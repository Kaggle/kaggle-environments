import 'react';

declare module 'react' {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface HTMLAttributes<T> {
    popover?: 'auto' | 'manual' | '';
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface ButtonHTMLAttributes<T> {
    popovertarget?: string;
    popovertargetaction?: 'show' | 'hide' | 'toggle';
  }
}
