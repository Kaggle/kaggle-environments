import { memo } from 'react';
import Meter from './Meter';
import GameBoard from './GameBoard';
import Legend from './Legend';
import Openings from './Openings';
import GameOver from './GameOver';

export default memo(function Layout() {
  return (
    <div id="renderer">
      <Meter />
      <GameBoard />
      <Legend />
      <Openings />
      <GameOver />
    </div>
  );
});
