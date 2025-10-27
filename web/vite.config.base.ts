import { defineConfig } from 'vite';
import checker from 'vite-plugin-checker';

export default defineConfig({
    base: './',
    optimizeDeps: {
        exclude: ['@kaggle-environments/core']
    },
    server: {
        host: '0.0.0.0',
        port: 5173,
        cors: true
    },
    plugins: [
        checker({ typescript: true }),
        {
            name: 'custom-header-plugin',
            configureServer(server) {
                const originalPrintUrls = server.printUrls;
                server.printUrls = () => {
                    const name = process.env.VITE_CUSTOM_HEADER_NAME;
                    const path = process.env.VITE_CUSTOM_HEADER_PATH;
                    if (name && path) {
                        const header = `\n  ┃ Running: ${name}\n  ┃ Path:    ${path}\n`;
                        process.stdout.write(header);
                    }
                    originalPrintUrls();
                };
            }
        }
    ]
});
