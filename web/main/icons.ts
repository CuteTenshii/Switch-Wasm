// Lucide icons as inline SVG, sized and hidden from assistive tech.

import { createElement, type IconNode } from 'lucide';

export function icon(node: IconNode): SVGElement {
  return createElement(node, { 'aria-hidden': 'true', 'width': 15, 'height': 15, 'stroke-width': 1.75 });
}
