const { execSync, spawn } = require('child_process');
const prompts = require('prompts');

// The command to run ('dev' or 'build')
const command = process.argv[2];
// An optional game name to target directly
const gameArg = process.argv[3];

if (!command) {
  console.error('Error: No command specified. Usage: node find-games.js <dev|build> [game-name]');
  process.exit(1);
}

// 1. Get all workspace packages from pnpm
let packages = [];
try {
  const pnpmOutput = execSync('pnpm m ls --json --depth -1', { encoding: 'utf8' });
  packages = JSON.parse(pnpmOutput);
} catch (e) {
  console.error('Error: Could not list pnpm workspaces. Make sure you are in a pnpm workspace root.');
  process.exit(1);
}

// 2. Filter for the game visualizer packages
const gameVisualizers = packages.filter(
  (pkg) => pkg.name && pkg.name.startsWith('@kaggle-environments/') && pkg.name.endsWith('-visualizer')
);

if (gameVisualizers.length === 0) {
  console.error('No game visualizers found. Make sure they are named like "@kaggle-environments/*-visualizer".');
  process.exit(0);
}

const path = require('path');

// ... (rest of the file is the same until runCommand)

const runCommand = (pkg) => {
  const packageName = pkg.name;
  const relativePath = path.relative(process.cwd(), pkg.path);

  console.log(`Running "${command}" for ${packageName}...`);
  // Use spawn with stdio: 'inherit' to preserve color and TTY features
  const child = spawn('pnpm', ['--filter', packageName, command], {
    stdio: 'inherit',
    // Use shell: true to ensure pnpm is found in the path, similar to exec's behavior
    shell: true,
    env: {
      ...process.env,
      VITE_CUSTOM_HEADER_NAME: packageName,
      VITE_CUSTOM_HEADER_PATH: relativePath,
    }
  });

  child.on('error', (err) => {
    console.error(`Failed to start command for ${packageName}:`, err);
  });

  child.on('exit', (code) => {
    if (code !== 0) {
      console.error(`\n'${packageName} ${command}' process exited with code ${code}`);
    }
  });
};

// 3. If a game name is passed as an argument, run it directly
if (gameArg) {
  if (gameArg === '--all') {
    console.log(`Running "${command}" for all ${gameVisualizers.length} visualizers...`);
    gameVisualizers.forEach(runCommand);
  } else {
    const targetPackage = gameVisualizers.find(pkg => pkg.name.includes(gameArg));
    if (targetPackage) {
      runCommand(targetPackage);
    } else {
      console.error(`Error: Could not find a visualizer package matching "${gameArg}".`);
      process.exit(1);
    }
  }
} else {
  // 4. Otherwise, show the interactive prompt
  (async () => {
    const response = await prompts({
      type: 'select',
      name: 'packageName',
      message: `Which game do you want to ${command}?`,
      choices: gameVisualizers.map(pkg => ({
        title: pkg.name,
        value: pkg.name
      })),
    });

    if (response.packageName) {
      const targetPackage = gameVisualizers.find(pkg => pkg.name === response.packageName);
      runCommand(targetPackage);
    } else {
      console.log('No game selected. Exiting.');
    }
  })();
}
