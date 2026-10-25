import { render } from 'preact';
import { App } from './ui/App';
import './styles.css';

// Register service worker for PWA
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      // Service worker registration failed — app still works
    });
  });
}

render(<App />, document.getElementById('app')!);
