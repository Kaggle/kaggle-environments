import { styled } from '@mui/material';
import * as React from 'react';
import Markdown from 'react-markdown';

const MarkdownContainer = styled('div')`
  color: ${(p) => p.theme.palette.text.primary};

  font-size: 14px;
  line-height: 22px;

  h1 {
    font-size: 20px;
    line-height: 24px;
    margin: 32px 0 24px;
  }

  h2 {
    font-size: 18px;
    line-height: 22px;
    margin: 24px 0 12px;
  }

  h3 {
    font-size: 16px;
    line-height: 20px;
    margin: 24px 0 8px;
  }

  h4,
  h5,
  h6 {
    font-size: 14px;
    line-height: 18px;
    margin: 24px 0 8px;
  }

  p {
    margin: 0 0 16px;
    /* This is required to show line breaks between paragraphs
     for markdown content that gets converted to HTML by showdown */
    /* https://stackoverflow.com/a/1409742 */
    br {
      content: ' ';
      display: block;
      height: 16px;
    }
  }

  /* If the header is the first element of the markdown, don't apply top margin */
  h1:first-child,
  h2:first-child,
  h3:first-child,
  h4:first-child,
  h5:first-child,
  h6:first-child {
    margin-top: 0;
  }

  ul,
  ol {
    padding-left: 16px;
  }

  ol {
    list-style-type: decimal;
  }

  ul {
    list-style-type: disc;
  }

  ul ol,
  ol ol {
    margin: 0;
    list-style-type: lower-latin;
  }

  ol ul,
  ul ul {
    list-style-type: circle;
  }

  li {
    margin: 8px 0;
    ol,
    ul {
      margin: 0 0 0 8px;
    }
  }
`;

export interface UserContentProps {
  markdown: string;
  style?: React.CSSProperties;
  className?: string;
}

export const UserContent = React.forwardRef<HTMLDivElement, UserContentProps>(({ markdown, style, className }, ref) => {
  return (
    <MarkdownContainer ref={ref} style={style} className={className}>
      <Markdown>{markdown}</Markdown>
    </MarkdownContainer>
  );
});

UserContent.displayName = 'UserContent';
