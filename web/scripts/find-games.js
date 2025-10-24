const { execSync, spawn } = require('child_process');
const prompts = require('prompts');
const path = require('path');

// The command to run ('dev' or 'build')
const command = process.argv[2];
// An optional game name to target directly (e.g., 'connectx' or '--all')
const gameArg = process.argv[3];

if (!command) {
    console.error('Error: No command specified. Usage: node find-games.js <dev|build> [game-name]');
    process.exit(1);
}

// Gets all workspace packages from pnpm
let packages = [];
try {
    const pnpmOutput = execSync('pnpm m ls --json --depth -1', { encoding: 'utf8' });
    packages = JSON.parse(pnpmOutput);
} catch (e) {
    console.error('Error: Could not list pnpm workspaces. Make sure you are in a pnpm workspace root.');
    process.exit(1);
}

// Filter for the game visualizer packages
const gameVisualizers = packages.filter(
    (pkg) => pkg.name && pkg.name.startsWith('@kaggle-environments/') && pkg.name.endsWith('-visualizer')
);

if (gameVisualizers.length === 0) {
    console.error('No game visualizers found. Make sure they are named like "@kaggle-environments/*-visualizer".');
    process.exit(0);
}

/**
 * Runs a command for a single package. Used for 'dev' or building one visualizer at a time.
 */
const runCommand = (pkg) => {
    const packageName = pkg.name;
    const relativePath = path.relative(process.cwd(), pkg.path);

    // Clear the screen only for the 'dev' command to mimic Vite's behavior
    if (command === 'dev') {
        console.clear();
    }

    console.log(`Running "pnpm ${command}" in ${relativePath}...`);

    const child = spawn('pnpm', [command], {
        stdio: 'inherit',
        shell: true, // Use shell: true to ensure pnpm is found in the path
        cwd: pkg.path,
        env: {
            ...process.env,
            VITE_CUSTOM_HEADER_NAME: packageName,
            VITE_CUSTOM_HEADER_PATH: relativePath
        }
    });

    child.on('error', (err) => {
        console.error(`Failed to start command for ${packageName}:`, err);
    });

    child.on('exit', (code) => {
        if (code !== 0) {
            console.error(`\n'${packageName} ${command}' process exited with code ${code}`);
            // Propagate the error code for CI/CD environments
            process.exit(code);
        }
    });
};

// Main logic to determine which command to run
if (gameArg) {
    if (command === 'build' && gameArg === '--all') {
        console.log(`Building all ${gameVisualizers.length} visualizers and their dependencies...`);
        const buildCommand = 'pnpm -r build --filter "*-visualizer"';
        console.log(`> ${buildCommand}\n`);

        try {
            execSync(buildCommand, { stdio: 'inherit' });
            console.log('\n✅ All visualizers built successfully.');
        } catch (error) {
            console.error('\n❌ Build failed.');
            process.exit(1);
        }
    } else if (gameArg === '--all') {
        console.log(`Running "${command}" for all ${gameVisualizers.length} visualizers...`);
        gameVisualizers.forEach(runCommand);
    } else {
        const targetPackage = gameVisualizers.find((pkg) => pkg.name.includes(gameArg));
        if (targetPackage) {
            runCommand(targetPackage);
        } else {
            console.error(`Error: Could not find a visualizer package matching "${gameArg}".`);
            process.exit(1);
        }
    }
} else {
    // If no game is specified, show the interactive prompt.
    (async () => {
        const response = await prompts({
            type: 'select',
            name: 'packageName',
            message: `Which game do you want to ${command}?`,
            choices: gameVisualizers.map((pkg) => ({
                title: pkg.name,
                value: pkg.name
            }))
        });

        if (response.packageName) {
            const targetPackage = gameVisualizers.find((pkg) => pkg.name === response.packageName);
            runCommand(targetPackage);
        } else {
            console.log('No game selected. Exiting.');
        }
    })();
}
