import { describe, expect, test } from 'bun:test';
import { extrasFor, holdDlc, holdUpdate, pairedNotes } from '../../web/main/pairing';

// The module keeps what it holds for the page's life, so each test pairs titles of its own.
const file = (name: string) => new File([], name);
const pack = (titleId: string, index: number) => ({ id: titleId + index, title_id: titleId, index });

describe('pairing', () => {
  test('an update goes to the launch of its own title only', () => {
    holdUpdate(file('patch.nsp'), { title: '0100aaaa00000000', version: '1.2.0' });
    expect(extrasFor('0100aaaa00000000').update?.version).toBe('1.2.0');
    expect(extrasFor('0100bbbb00000000').update).toBeNull();
  });

  test('a newer update takes the place of the one held', () => {
    holdUpdate(file('old.nsp'), { title: '0100cccc00000000', version: '1.0.0' });
    holdUpdate(file('new.nsp'), { title: '0100dddd00000000', version: '2.0.0' });
    expect(extrasFor('0100cccc00000000').update).toBeNull();
    expect(extrasFor('0100dddd00000000').update?.file.name).toBe('new.nsp');
  });

  test('an update held for another title is told under the open one, with the title it is for', () => {
    holdUpdate(file('other.nsp'), { title: '0100eeee00000000', version: '' });
    const notes = pairedNotes('0100ffff00000000');
    expect(notes.some((line) => line.includes('0100eeee00000000') && line.includes('other.nsp'))).toBe(true);
  });

  test('add-on packs add up per title, and a file dropped again replaces itself', () => {
    const title = '0100abab00000000';
    holdDlc(file('a.nsp'), [pack(title, 1), pack(title, 2)]);
    holdDlc(file('b.nsp'), [pack(title, 3)]);
    holdDlc(file('a.nsp'), [pack(title, 1)]);
    expect(extrasFor(title).packs).toBe(2);
    expect(extrasFor('0100cdcd00000000').packs).toBe(0);
  });

  test('add-on content held for another title is told with the title it is for', () => {
    holdDlc(file('extra.nsp'), [pack('0100efef00000000', 1)]);
    expect(pairedNotes('0100fefe00000000').some((line) => line.includes('0100efef00000000'))).toBe(true);
    expect(pairedNotes('0100efef00000000').some((line) => line.includes('extra.nsp'))).toBe(true);
  });
});
