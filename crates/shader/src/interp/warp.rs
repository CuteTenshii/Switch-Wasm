//! Completing shuffles and votes across a warp.

use super::*;

/// Hardware warp width, which `shfl` clamps and segment masks are written against.
pub const WARP_LANES: usize = 32;

/// Complete every shuffle and vote a warp (in lane order) is suspended on, reading
/// all sources before writing. A shuffle to a lane the clamp allows but the warp
/// lacks reads the caller's own value. Votes count only lanes that reached them.
pub fn resolve_warp(warp: &mut [Invocation]) {
    let requests: Vec<Option<Exchange>> = warp
        .iter_mut()
        .map(|invocation| invocation.exchange.take())
        .collect();
    let (mut voters, mut ballot) = (0u32, 0u32);
    for (lane, request) in requests.iter().enumerate() {
        if let Some(Exchange::Vote(vote)) = request {
            voters |= 1 << lane;
            ballot |= u32::from(vote.holds) << lane;
        }
    }
    let answers: Vec<Option<(u8, u8, u32, bool)>> = requests
        .iter()
        .enumerate()
        .map(|(lane, request)| match *request {
            None => None,
            Some(Exchange::Shuffle(shuffle)) => {
                let (from, in_bounds) = shuffle_source(shuffle, lane as u32);
                let value = match warp.get(from as usize).filter(|_| in_bounds) {
                    Some(peer) => peer.reg(shuffle.src),
                    None => warp[lane].reg(shuffle.src),
                };
                Some((shuffle.dst, shuffle.pred, value, in_bounds))
            }
            Some(Exchange::Vote(vote)) => {
                let verdict = match vote.mode {
                    VoteMode::All => ballot == voters,
                    VoteMode::Any => ballot != 0,
                    VoteMode::Eq => ballot == 0 || ballot == voters,
                };
                Some((vote.dst, vote.pred, ballot, verdict))
            }
        })
        .collect();
    for (invocation, answer) in warp.iter_mut().zip(answers) {
        let Some((dst, pred, value, flag)) = answer else {
            continue;
        };
        invocation.set_reg(dst, value);
        invocation.set_pred(pred, flag);
    }
}

/// Which lane `shuffle` reads from `lane`, and whether it was within bounds.
fn shuffle_source(shuffle: Shuffle, lane: u32) -> (u32, bool) {
    let clamp = (shuffle.mask & 0x1f) as i32;
    let segment = ((shuffle.mask >> 8) & 0x1f) as i32;
    let lane = lane as i32;
    let index = shuffle.index as i32;
    let floor = lane & segment;
    let ceiling = floor | (clamp & !segment);
    let (from, in_bounds) = match shuffle.mode {
        ShflMode::Idx => {
            let from = (index & !segment) | floor;
            (from, from <= ceiling)
        }
        // `up` is bounded from below by the segment floor.
        ShflMode::Up => {
            let from = lane - index;
            (from, from >= ceiling)
        }
        ShflMode::Down => {
            let from = lane + index;
            (from, from <= ceiling)
        }
        ShflMode::Bfly => {
            let from = lane ^ index;
            (from, from <= ceiling)
        }
    };
    (from.max(0) as u32, in_bounds && from >= 0)
}
