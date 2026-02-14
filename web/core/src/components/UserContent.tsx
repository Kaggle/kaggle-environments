import * as React from 'react';
import Markdown from 'react-markdown';

export interface UserContentProps {
  markdown: string;
  style?: React.CSSProperties;
  className?: string;
}

export const UserContent = React.forwardRef<HTMLDivElement, UserContentProps>(({ markdown, style, className }, ref) => {
  return (
    <div ref={ref} style={{ fontFamily: 'Inter, sans-serif', ...style }} className={className}>
      <Markdown>{markdown}</Markdown>
    </div>
  );
});

UserContent.displayName = 'UserContent';
