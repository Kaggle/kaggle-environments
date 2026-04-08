import { memo } from 'react';
import StyledBoard from './StyledBoard';
import Legend from './Legend';
import Meter from './Meter';
import Openings from './Openings';
import GameOver from './GameOver';

export default memo(function Layout() {
  return (
    <div id="renderer">
      <Meter />
      <StyledBoard />
      <Legend />
      <Openings />
      <GameOver />
    </div>
  );
});
