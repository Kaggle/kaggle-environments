#!/usr/bin/env node
/**
 * Validates that the build/manifest.json file is properly populated.
 * This ensures the build pipeline correctly generates artifact metadata.
 *
 * Exit codes:
 *   0 - Manifest is valid
 *   1 - Manifest is missing, empty, or malformed
 */

const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.resolve(__dirname, '../..');
const MANIFEST_PATH = path.join(ROOT_DIR, 'build', 'manifest.json');
const BUILD_DIR = path.join(ROOT_DIR, 'build');

function fail(message) {
  console.error(`❌ Manifest validation failed: ${message}`);
  process.exit(1);
}

function validateManifest() {
  console.log('--- Validating manifest.json ---');

  // Check build directory exists
  if (!fs.existsSync(BUILD_DIR)) {
    fail(`Build directory does not exist: ${BUILD_DIR}`);
  }

  // Check manifest file exists
  if (!fs.existsSync(MANIFEST_PATH)) {
    fail(`Manifest file does not exist: ${MANIFEST_PATH}`);
  }

  // Read and parse manifest
  let manifestContent;
  try {
    manifestContent = fs.readFileSync(MANIFEST_PATH, 'utf8');
  } catch (err) {
    fail(`Could not read manifest file: ${err.message}`);
  }

  // Check manifest is not empty
  if (!manifestContent.trim()) {
    fail('Manifest file is empty');
  }

  // Parse JSON
  let manifest;
  try {
    manifest = JSON.parse(manifestContent);
  } catch (err) {
    fail(`Manifest is not valid JSON: ${err.message}`);
  }

  // Check manifest is an object
  if (typeof manifest !== 'object' || manifest === null || Array.isArray(manifest)) {
    fail('Manifest must be a JSON object');
  }

  // Check manifest has at least one game entry
  const gameNames = Object.keys(manifest);
  if (gameNames.length === 0) {
    fail('Manifest contains no games - build may have failed silently');
  }

  // Validate each game entry
  for (const gameName of gameNames) {
    const visualizers = manifest[gameName];

    // Each game must map to an array
    if (!Array.isArray(visualizers)) {
      fail(`Game "${gameName}" must map to an array of visualizer names`);
    }

    // Each game must have at least one visualizer
    if (visualizers.length === 0) {
      fail(`Game "${gameName}" has no visualizers`);
    }

    // Each visualizer must be a non-empty string
    for (const viz of visualizers) {
      if (typeof viz !== 'string' || !viz.trim()) {
        fail(`Game "${gameName}" has invalid visualizer entry: ${JSON.stringify(viz)}`);
      }
    }

    // Verify the visualizer directory actually exists in build/
    for (const viz of visualizers) {
      const vizDir = path.join(BUILD_DIR, gameName, viz);
      if (!fs.existsSync(vizDir)) {
        fail(`Visualizer directory missing for ${gameName}/${viz}: ${vizDir}`);
      }
    }
  }

  // Success
  console.log(`✅ Manifest validation passed`);
  console.log(`   Found ${gameNames.length} game(s):`);
  for (const gameName of gameNames.sort()) {
    console.log(`     - ${gameName}: [${manifest[gameName].join(', ')}]`);
  }
}

validateManifest();
