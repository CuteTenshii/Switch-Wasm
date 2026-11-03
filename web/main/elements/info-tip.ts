// <info-tip text="...">: an info button whose popover shows the text on hover, focus or click.

import { Info } from 'lucide';
import { el } from '../dom';
import { icon } from '../icons';

const GAP = 6;
const MARGIN = 8;
let nextId = 0;

class InfoTip extends HTMLElement {
  static observedAttributes = ['text'];

  private readonly button = el('button', 'info-tip-button');
  private readonly bubble = el('div', 'info-tip-bubble');

  connectedCallback(): void {
    if (this.button.isConnected) return;
    this.bubble.id = `info-tip-${nextId++}`;
    this.bubble.setAttribute('popover', 'manual');
    this.bubble.setAttribute('role', 'tooltip');
    this.button.type = 'button';
    this.button.setAttribute('aria-label', 'More information');
    this.button.setAttribute('aria-describedby', this.bubble.id);
    this.button.append(icon(Info));
    this.append(this.button, this.bubble);
    this.syncText();

    const show = (): void => {
      if (!this.bubble.textContent) return;
      this.bubble.showPopover();
      const anchor = this.button.getBoundingClientRect();
      const width = this.bubble.offsetWidth;
      const left = Math.min(anchor.left, window.innerWidth - width - MARGIN);
      this.bubble.style.left = `${Math.max(MARGIN, left)}px`;
      this.bubble.style.top = `${anchor.bottom + GAP}px`;
    };
    const hide = (): void => {
      if (this.bubble.matches(':popover-open')) this.bubble.hidePopover();
    };
    this.button.addEventListener('pointerenter', show);
    this.button.addEventListener('pointerleave', hide);
    this.button.addEventListener('focus', show);
    this.button.addEventListener('blur', hide);
    this.button.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') hide();
    });
  }

  attributeChangedCallback(): void {
    this.syncText();
  }

  private syncText(): void {
    const text = this.getAttribute('text') ?? '';
    this.bubble.textContent = text;
    this.hidden = !text;
  }
}

customElements.define('info-tip', InfoTip);
