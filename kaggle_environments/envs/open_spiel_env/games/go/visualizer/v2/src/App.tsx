import Controls from './components/Controls';
import StyledGoboard from './components/StyledGoboard';
import './App.css';

function App() {
  return (
    <div className="container">
      <StyledGoboard />
      <Controls />
    </div>
  );
}

export default App;
