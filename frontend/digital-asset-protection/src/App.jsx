import OfficialAssetUploader from './components/OfficialAssetUploader';
import SuspectVideoUploader from './components/SuspectVideoUploader';
import './App.css';

function App() {
  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>Detection Pipeline Control Center</h1>
        <p>Manage Reference Assets & Simulate Piracy Ingestion</p>
      </header>
      
      <main className="dashboard-main">
        {/* Left Side: The Shield (Official Masters) */}
        <OfficialAssetUploader />
        
        {/* Right Side: The Sword (Suspect Ingestion) */}
        <SuspectVideoUploader />
      </main>
    </div>
  );
}

export default App;