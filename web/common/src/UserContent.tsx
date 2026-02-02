import * as React from "react";
import Markdown from "react-markdown";

export interface UserContentProps {
  markdown: string;
  youtubePreviewOnly?: boolean;
  style?: React.CSSProperties;
  className?: string;
}

export const UserContent = React.forwardRef<HTMLDivElement, UserContentProps>(
  ({ markdown, youtubePreviewOnly: _youtubePreviewOnly, style, className }, ref) => {
    return (
      <div ref={ref} style={style} className={className}>
        <Markdown>{markdown}</Markdown>
      </div>
    );
  }
);

UserContent.displayName = "UserContent";
