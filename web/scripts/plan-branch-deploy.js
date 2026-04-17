#!/usr/bin/env node
/**
 * Discovers which visualizers the current branch owns (by reading
 * branch-owned.json from master), builds them, and stages artifacts
 * for deployment.
 *
 * This script is meant to run in a Cloud Build step. It reads the
 * trusted branch-owned.json from the master branch (not from the
 * branch being built) so that branches cannot escalate their own
 * deploy permissions.
 *
 * Output:
 *   /workspace/branch-deploy/{game}/{viz}/  — built dist files
 *   /workspace/branch-deploy/plan.json      — deploy plan for next step
 */
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const branch = process.env.BRANCH_NAME;
if (!branch) {
  console.log('BRANCH_NAME not set, skipping.');
  process.exit(0);
}

// Read branch-owned.json from master (trusted source, not the branch being built)
let configJson;
try {
  execSync('git fetch origin master --depth=1', { stdio: 'inherit' });
  configJson = execSync('git show origin/master:web/config/branch-owned.json', { encoding: 'utf8' });
} catch {
  console.log('Could not read branch-owned.json from master, skipping.');
  process.exit(0);
}

let config;
try {
  config = JSON.parse(configJson);
} catch {
  console.log('branch-owned.json is not valid JSON, skipping.');
  process.exit(0);
}

// Find visualizers that master says this branch owns
const owned = Object.entries(config)
  .filter(([, ownerBranch]) => ownerBranch === branch)
  .map(([key]) => {
    const [game, viz] = key.split('/');
    return { game, viz };
  });

if (owned.length === 0) {
  console.log(`No visualizers owned by branch '${branch}', nothing to do.`);
  process.exit(0);
}

console.log(`Branch '${branch}' owns ${owned.length} visualizer(s):`);
owned.forEach(({ game, viz }) => console.log(`  - ${game}/${viz}`));

/**
 * Derive the filesystem path for a visualizer from game/visualizer names.
 * Follows the repo convention:
 *   - open_spiel_* games: kaggle_environments/envs/open_spiel_env/games/{name}/visualizer/{viz}
 *   - regular games:      kaggle_environments/envs/{game}/visualizer/{viz}
 */
function getVisualizerPath(game, viz) {
  if (game.startsWith('open_spiel_')) {
    const inner = game.slice('open_spiel_'.length);
    return `kaggle_environments/envs/open_spiel_env/games/${inner}/visualizer/${viz}`;
  }
  return `kaggle_environments/envs/${game}/visualizer/${viz}`;
}

// Build core dependencies once
console.log('\nBuilding core dependencies...');
execSync('pnpm --filter @kaggle-environments/core build', { stdio: 'inherit' });
execSync('pnpm --filter @kaggle-environments/common build', {
  stdio: 'inherit',
});

const stageDir = '/workspace/branch-deploy';
fs.mkdirSync(stageDir, { recursive: true });

const plan = [];

for (const { game, viz } of owned) {
  const vizPath = getVisualizerPath(game, viz);

  // Skip if this branch hasn't created the visualizer yet
  const pkgJsonPath = path.join(vizPath, 'package.json');
  if (!fs.existsSync(pkgJsonPath)) {
    console.log(
      `\nWARNING: ${game}/${viz} registered in branch-owned.json but ` +
        `${vizPath} does not exist on this branch. Skipping.`
    );
    continue;
  }

  const pkgJson = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));
  const pkgName = pkgJson.name;

  console.log(`\nBuilding ${pkgName} (${game}/${viz})...`);
  execSync(`pnpm --filter ${pkgName} build`, { stdio: 'inherit' });

  const destDir = path.join(stageDir, game, viz);
  fs.mkdirSync(destDir, { recursive: true });
  execSync(`cp -r ${vizPath}/dist/* ${destDir}/`);

  plan.push({ game, viz });
}

// Write deploy plan for the next step
fs.writeFileSync(path.join(stageDir, 'plan.json'), JSON.stringify(plan, null, 2));

console.log(`\nBuild complete. ${plan.length} visualizer(s) staged for deploy.`);
