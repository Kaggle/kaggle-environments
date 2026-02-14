import * as React from 'react';
import { styled } from '@mui/material/styles';
import { List } from '@mui/material';

const SpacedListItem = styled('li')`
  /* Reset default li styles just in case */
  margin: 0;
  padding: 0;
  list-style: none;

  /* Apply the gap here.
    Virtuoso measures this element, so it will account for the extra space.
    WARNING: if this is margin instead of padding it will cause Jitter
    */
  padding-top: 20px;
  box-sizing: border-box;

  ${({ theme }) => theme.breakpoints.down('tablet')} {
    padding-top: 8px;
  }
`;

const StepList = styled(List)`
  && {
    margin: 0;
    padding: 16px 24px;
  }
`;

/* Hide all scrollbars- they get distracting when content is constantly scrolling */
const LogsScroller = styled('div')`
  scrollbar-width: none;
  -ms-overflow-style: none;
  overflow-anchor: none;
  scrollbar-gutter: stable;

  &::-webkit-scrollbar {
    display: none;
    width: 0;
    background: transparent;
  }
`;

export const VirtuosoScrollerList = React.forwardRef((props, ref: React.ForwardedRef<any>) => {
  return <StepList {...props} ref={ref} />;
});

export const VirtuosoListItem = React.forwardRef((props, ref: React.ForwardedRef<any>) => {
  return <SpacedListItem {...props} ref={ref} />;
});

export const Scroller = React.forwardRef(({ ...props }, ref: React.ForwardedRef<any>) => (
  <LogsScroller ref={ref} {...props} />
));
