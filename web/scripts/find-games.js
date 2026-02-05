/* eslint-disable @typescript-eslint/no-require-imports */
const { execSync, spawn } = require('child_process');
const prompts = require('prompts');
const path = require('path');

// Parse arguments
const args = process.argv.slice(2);
const command = args[0]; // 'dev', 'build', or 'test-server'

// Parse flags and positional arguments
let gameArg = null;
let withReplay = false;

for (let i = 1; i < args.length; i++) {
  if (args[i] === '--with-replay') {
    withReplay = true;
  } else if (!args[i].startsWith('--')) {
    gameArg = args[i];
  }
}

if (!command) {
  console.error(
    'Error: No command specified. Usage: node find-games.js <dev|build|test-server> [game-name] [--with-replay]'
  );
  process.exit(1);
}

// Determine the actual npm script to run
const getDevScript = () => {
  if (withReplay || command === 'test-server') {
    return 'dev-with-replay';
  }
  return 'dev';
};

// Gets all workspace packages from pnpm
let packages = [];
try {
  const pnpmOutput = execSync('pnpm m ls --json --depth -1', { encoding: 'utf8' });
  packages = JSON.parse(pnpmOutput);
} catch {
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
 * Runs a command for a package.
 * - For 'dev', it first builds all dependencies, then runs dev servers in parallel.
 * - For 'build', it runs the command only for the selected package.
 */
const runCommand = (pkg) => {
  const packageName = pkg.name;
  const relativePath = path.relative(process.cwd(), pkg.path);

  // Clear the screen only for 'dev' to mimic Vite's behavior
  if (command === 'dev') {
    console.clear();
  }

  let cmdToRun, cmdArgs, cwd;

  if (command === 'dev' || command === 'test-server') {
    const devScript = getDevScript();
    try {
      // STEP 1: Build all dependencies of the target package first.
      // The `...^` syntax targets all dependencies, but NOT the package itself.
      console.log(`[1/2] Building dependencies for ${packageName}...`);
      execSync(`pnpm --filter ${packageName}...^ build`, {
        stdio: 'inherit',
        cwd: process.cwd(),
      });
      console.log(`✅ Dependencies built successfully.`);
    } catch {
      console.error('\n❌ Initial build of dependencies failed. Aborting.');
      process.exit(1);
    }

    // STEP 2: Now, run the dev server from the monorepo root.
    // For test-server/with-replay, run only the target package with dev-with-replay.
    // For regular dev, run parallel dev and watch commands.
    if (devScript === 'dev-with-replay') {
      console.log(`\n[2/2] Starting test server for ${packageName} with replay...`);
      cmdToRun = 'pnpm';
      cmdArgs = ['--filter', packageName, devScript];
      cwd = process.cwd();
    } else {
      console.log(`\n[2/2] Starting dev servers for ${packageName} and its dependencies...`);
      cmdToRun = 'pnpm';
      cmdArgs = ['--parallel', '--filter', `${packageName}...`, 'dev'];
      cwd = process.cwd();
    }
  } else {
    // For 'build' of a single package, the original logic is fine.
    console.log(`Running "pnpm ${command}" in ${relativePath}...`);
    cmdToRun = 'pnpm';
    cmdArgs = [command];
    cwd = pkg.path; // Run inside the specific package directory
  }

  const child = spawn(cmdToRun, cmdArgs, {
    stdio: 'inherit',
    shell: true,
    cwd: cwd,
    env: {
      ...process.env,
      VITE_CUSTOM_HEADER_NAME: packageName,
      VITE_CUSTOM_HEADER_PATH: relativePath,
    },
  });

  child.on('error', (err) => {
    console.error(`Failed to start command for ${packageName}:`, err);
  });

  child.on('exit', (code) => {
    if (code !== 0) {
      console.error(`\n'${packageName} ${command}' process exited with code ${code}`);
      process.exit(code);
    }
  });
};

// Main logic to determine which command to run
if (gameArg) {
  if (command === 'build' && gameArg === '--all') {
    console.log(`Building all ${gameVisualizers.length} visualizers and their dependencies...`);
    try {
      // 1. Build Core first
      console.log('--- Step 1: Building @kaggle-environments/core ---');
      execSync('pnpm --filter @kaggle-environments/core build', { stdio: 'inherit' });

      // 2. Build Common second
      console.log('\n--- Step 2: Building @kaggle-environments/common ---');
      execSync('pnpm --filter @kaggle-environments/common build', { stdio: 'inherit' });

      // 3. Build all visualizers in parallel (since dependencies are now ready)
      console.log('\n--- Step 3: Building all visualizers ---');
      const buildCommand = 'pnpm -r --parallel build --filter "*-visualizer"';
      execSync(buildCommand, { stdio: 'inherit' });

      console.log('\n✅ All visualizers and dependencies built successfully.');
    } catch {
      console.error('\n❌ Build failed during sequential execution.');
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
        value: pkg.name,
      })),
    });

    if (response.packageName) {
      const targetPackage = gameVisualizers.find((pkg) => pkg.name === response.packageName);
      runCommand(targetPackage);
    } else {
      console.log('No game selected. Exiting.');
    }
  })();
}
