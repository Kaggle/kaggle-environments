import StyledChessboard from './components/StyledChessboard';
import Controls from './components/Controls';
import Legend from './components/Legend';
import Openings from './components/Openings';
import './App.css';

function App() {
  return (
    <div className="container">
      <StyledChessboard />
      <Controls />
      <Legend />
      <Openings />
    </div>
  );
}

export default App;
