import 'react';

declare module 'react' {
  interface HTMLAttributes<T> {
    popover?: 'auto' | 'manual' | '';
  }

  interface ButtonHTMLAttributes<T> {
    popovertarget?: string;
    popovertargetaction?: 'show' | 'hide' | 'toggle';
  }
}
