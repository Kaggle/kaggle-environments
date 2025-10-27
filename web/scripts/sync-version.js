const fs = require('fs');
const path = require('path');

const pyprojectPath = path.resolve(__dirname, '../../pyproject.toml');
const packageJsonPath = path.resolve(__dirname, '../core/package.json');

try {
  // Read pyproject.toml
  const pyprojectContent = fs.readFileSync(pyprojectPath, 'utf8');
  const versionMatch = pyprojectContent.match(/^version\s*=\s*"(.+?)"/m);

  if (!versionMatch) {
    throw new Error('Could not find version in pyproject.toml');
  }
  const version = versionMatch[1];
  console.log(`Found version ${version} in pyproject.toml`);

  // Read package.json
  const packageJsonContent = fs.readFileSync(packageJsonPath, 'utf8');
  const packageJson = JSON.parse(packageJsonContent);

  // Update version
  packageJson.version = version;

  // Write back to package.json
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2) + '\n');
  console.log(`Successfully updated version in web/core/package.json to ${version}`);

} catch (error) {
  console.error('Error syncing version:', error.message);
  process.exit(1);
}
