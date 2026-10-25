import { render } from 'preact';
import { App } from './ui/App';
import './styles.css';

// Service worker is registered automatically by vite-plugin-pwa (injectRegister: 'auto')

render(<App />, document.getElementById('app')!);
