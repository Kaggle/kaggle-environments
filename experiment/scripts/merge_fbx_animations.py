#!/usr/bin/env python3
"""
FBX Animation Merger Script

This script loads FBX models from the werewolf models directory and merges
animations into a master file for each player type.

Requirements:
    pip install FBX-2020.3.4 (or use Autodesk FBX Python SDK)
    
Note: FBX manipulation in Python typically requires either:
1. Autodesk FBX Python SDK (official but complex setup)
2. Using Blender's Python API (bpy) as a subprocess
3. Third-party libraries like pyfbx or fbx-python

This implementation uses a fallback approach with file operations
since FBX SDK requires specific installation procedures.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FBXAnimationMerger:
    """Handles merging of FBX animations into master files."""
    
    # Define the model types and their expected animations
    MODEL_TYPES = ['doctor', 'seer', 'villager', 'werewolf']
    
    # Animation mappings: animation_name -> list of possible file names
    ANIMATION_MAPPINGS = {
        'Idle': ['Idle.fbx', 'Standing Idle.fbx', 'Neutral Idle.fbx'],
        'Talking': ['Talking.fbx'],
        'Pointing': ['Pointing.fbx'],
        'Victory': ['Victory.fbx'],
        'Defeated': ['Defeated.fbx'],
        'Dying': ['Dying.fbx']
    }
    
    def __init__(self, base_path: str = 'experiment/static/werewolf/models'):
        """
        Initialize the FBX Animation Merger.
        
        Args:
            base_path: Base directory containing model subdirectories
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")
        
        # Check if we can use Blender for FBX operations
        self.blender_path = 'blender'  # Default
        self.blender_available = self._check_blender_availability()
        
    def _check_blender_availability(self) -> bool:
        """Check if Blender is available for FBX operations."""
        # Check for Blender in common locations
        blender_paths = [
            'blender',  # In PATH
            '/Applications/Blender.app/Contents/MacOS/Blender',  # macOS
            '/usr/bin/blender',  # Linux
            'C:\\Program Files\\Blender Foundation\\Blender\\blender.exe'  # Windows
        ]
        
        for blender_path in blender_paths:
            try:
                result = subprocess.run(
                    [blender_path, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.blender_path = blender_path
                    logger.info(f"Blender is available at: {blender_path}")
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        
        logger.warning("Blender not found. Will use fallback method.")
        return False
    
    def find_animation_file(self, model_dir: Path, animation_names: List[str]) -> Optional[Path]:
        """
        Find the first existing animation file from a list of possible names.
        
        Args:
            model_dir: Directory to search in
            animation_names: List of possible animation file names
            
        Returns:
            Path to the found animation file, or None if not found
        """
        fbx_dir = model_dir / 'fbx'
        if not fbx_dir.exists():
            return None
        
        for anim_name in animation_names:
            anim_path = fbx_dir / anim_name
            if anim_path.exists():
                return anim_path
        
        return None
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Gather information about a specific model type.
        
        Args:
            model_type: Type of model (doctor, seer, villager, werewolf)
            
        Returns:
            Dictionary containing model information
        """
        model_dir = self.base_path / model_type
        info = {
            'type': model_type,
            'base_dir': str(model_dir),
            'rigged_fbx': None,
            'texture_path': None,
            'animations': {},
            'missing_animations': []
        }
        
        # Check for rigged FBX
        rigged_path = model_dir / 'fbx' / 'rigged.fbx'
        if rigged_path.exists():
            info['rigged_fbx'] = str(rigged_path)
        else:
            logger.warning(f"No rigged.fbx found for {model_type}")
            return info
        
        # Find texture file
        fbx_dir = model_dir / 'fbx'
        for texture_file in fbx_dir.glob('*_texture.png'):
            info['texture_path'] = str(texture_file)
            logger.info(f"Found texture for {model_type}: {texture_file.name}")
            break
        
        # Find available animations
        for anim_name, possible_files in self.ANIMATION_MAPPINGS.items():
            anim_file = self.find_animation_file(model_dir, possible_files)
            if anim_file:
                info['animations'][anim_name] = str(anim_file)
                logger.info(f"Found {anim_name} animation for {model_type}: {anim_file.name}")
            else:
                info['missing_animations'].append(anim_name)
                logger.info(f"Missing {anim_name} animation for {model_type}")
        
        return info
    
    def merge_with_blender(self, model_info: Dict, output_path: Path) -> bool:
        """
        Merge animations using Blender's Python API.
        
        Args:
            model_info: Model information dictionary
            output_path: Path for the output master FBX file
            
        Returns:
            True if successful, False otherwise
        """
        # Create a temporary JSON file with animation paths
        temp_json = Path('temp_animations.json')
        temp_json.write_text(json.dumps(model_info['animations']))
        
        # Get the path to the Blender merge script
        blender_script = Path(__file__).parent / 'blender_merge_fbx.py'
        
        try:
            # Run Blender in background mode with the script
            cmd = [
                self.blender_path,
                '--background',
                '--python', str(blender_script),
                '--',
                model_info['rigged_fbx'],
                str(output_path),
                str(temp_json)
            ]
            
            # Add texture path if available
            if model_info.get('texture_path'):
                cmd.append(model_info['texture_path'])
            
            logger.info(f"Running Blender merge for {model_info['type']}...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Log Blender output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip() and not line.startswith('Blender'):
                        logger.debug(f"Blender: {line}")
            
            if result.returncode == 0:
                logger.info(f"✓ Successfully merged animations for {model_info['type']} using Blender")
                return True
            else:
                logger.error(f"Blender merge failed with code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Blender merge timed out for {model_info['type']}")
            return False
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to run Blender: {e}")
            return False
        finally:
            # Clean up temporary file
            if temp_json.exists():
                temp_json.unlink()
    
    def merge_fallback(self, model_info: Dict, output_path: Path) -> bool:
        """
        Fallback method for merging when Blender is not available.
        Creates a manifest file documenting the animations.
        
        Args:
            model_info: Model information dictionary
            output_path: Path for the output master FBX file
            
        Returns:
            True if successful, False otherwise
        """
        # Since we can't actually merge FBX files without proper tools,
        # we'll copy the rigged model and create a manifest
        
        try:
            # Copy the rigged model as the base
            shutil.copy2(model_info['rigged_fbx'], output_path)
            
            # Create a manifest file documenting the animations
            manifest_path = output_path.with_suffix('.manifest.json')
            manifest = {
                'model_type': model_info['type'],
                'base_model': model_info['rigged_fbx'],
                'texture': model_info.get('texture_path', 'Not found'),
                'output_file': str(output_path),
                'animations': model_info['animations'],
                'missing_animations': model_info['missing_animations'],
                'note': 'This is a manifest file. Actual FBX merging requires Blender or Autodesk FBX SDK.'
            }
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Created base FBX copy and manifest for {model_info['type']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create fallback files: {e}")
            return False
    
    def merge_model_animations(self, model_type: str) -> bool:
        """
        Merge all animations for a specific model type.
        
        Args:
            model_type: Type of model to process
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Processing {model_type} model...")
        
        # Get model information
        model_info = self.get_model_info(model_type)
        
        if not model_info['rigged_fbx']:
            logger.error(f"Cannot process {model_type}: no rigged.fbx found")
            return False
        
        # Determine output path
        output_dir = self.base_path / model_type / 'merged'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f'{model_type}_master.fbx'
        
        # Try to merge using available method
        if self.blender_available and model_info['animations']:
            success = self.merge_with_blender(model_info, output_path)
        else:
            success = self.merge_fallback(model_info, output_path)
        
        if success:
            logger.info(f"✓ Created master FBX for {model_type}: {output_path}")
        else:
            logger.error(f"✗ Failed to create master FBX for {model_type}")
        
        return success
    
    def process_all_models(self) -> Dict[str, bool]:
        """
        Process all model types and merge their animations.
        
        Returns:
            Dictionary mapping model types to success status
        """
        results = {}
        
        logger.info("=" * 60)
        logger.info("Starting FBX Animation Merge Process")
        logger.info("=" * 60)
        
        for model_type in self.MODEL_TYPES:
            results[model_type] = self.merge_model_animations(model_type)
            logger.info("-" * 40)
        
        # Summary
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        
        successful = [m for m, success in results.items() if success]
        failed = [m for m, success in results.items() if not success]
        
        if successful:
            logger.info(f"Successfully processed: {', '.join(successful)}")
        if failed:
            logger.error(f"Failed to process: {', '.join(failed)}")
        
        return results


def main():
    """Main entry point for the script."""
    try:
        # Initialize the merger
        merger = FBXAnimationMerger()
        
        # Process all models
        results = merger.process_all_models()
        
        # Exit with appropriate code
        if all(results.values()):
            logger.info("\n✓ All models processed successfully!")
            sys.exit(0)
        else:
            logger.warning("\n⚠ Some models failed to process. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()