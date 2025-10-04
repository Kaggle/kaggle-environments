#!/usr/bin/env python3
"""
Test script to verify FBX animations using Blender.
This script opens Blender with the merged FBX to inspect animations.
"""

import subprocess
import sys
from pathlib import Path

def test_fbx_in_blender(fbx_path):
    """Open an FBX file in Blender for inspection."""
    blender_path = '/Applications/Blender.app/Contents/MacOS/Blender'
    
    # Create a test script for Blender
    test_script = """
import bpy
import sys

# Clear the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import the FBX
filepath = sys.argv[-1]
print(f"\\nImporting FBX: {filepath}")
print("=" * 60)

bpy.ops.import_scene.fbx(
    filepath=filepath,
    use_custom_normals=True,
    use_image_search=True,
    use_anim=True,
    use_custom_props=True,
    ignore_leaf_bones=False,
    automatic_bone_orientation=False,
    primary_bone_axis='Y',
    secondary_bone_axis='X'
)

# Find armature and list animations
armature = None
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        armature = obj
        break

if armature:
    print(f"\\nFound armature: {armature.name}")
    
    if armature.animation_data:
        print("\\nAnimation Data:")
        print(f"  Active action: {armature.animation_data.action.name if armature.animation_data.action else 'None'}")
        
        # List all actions
        print(f"\\nAll Actions in file ({len(bpy.data.actions)}):")
        for action in bpy.data.actions:
            print(f"  - {action.name}: {len(action.fcurves)} curves, frame range: {action.frame_range}")
        
        # List NLA tracks
        if armature.animation_data.nla_tracks:
            print(f"\\nNLA Tracks ({len(armature.animation_data.nla_tracks)}):")
            for track in armature.animation_data.nla_tracks:
                print(f"  - Track: {track.name}")
                for strip in track.strips:
                    print(f"    - Strip: {strip.name}, Action: {strip.action.name if strip.action else 'None'}")
                    if strip.action:
                        print(f"      Frame range: {strip.action.frame_range}")
        else:
            print("\\nNo NLA tracks found")
    else:
        print("\\nNo animation data found on armature")
else:
    print("\\nNo armature found in the file")

# List all objects
print(f"\\nAll objects in scene ({len(bpy.context.scene.objects)}):")
for obj in bpy.context.scene.objects:
    print(f"  - {obj.name} ({obj.type})")

print("\\n" + "=" * 60)
print("Test complete. You can now inspect the file in Blender.")
"""
    
    # Save the test script
    script_path = Path('temp_test_fbx.py')
    script_path.write_text(test_script)
    
    try:
        # Run Blender with the test script
        cmd = [blender_path, '--python', str(script_path), '--', str(fbx_path)]
        print(f"Opening {fbx_path} in Blender for inspection...")
        subprocess.run(cmd)
    finally:
        # Clean up
        if script_path.exists():
            script_path.unlink()

def main():
    """Main function to test FBX files."""
    if len(sys.argv) < 2:
        # Test all merged files
        base_path = Path('experiment/static/werewolf/models')
        models = ['doctor', 'seer', 'villager', 'werewolf']
        
        print("Select a model to test:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        choice = input("Enter number (1-4): ")
        try:
            model = models[int(choice) - 1]
            fbx_path = base_path / model / 'merged' / f'{model}_master.fbx'
            if fbx_path.exists():
                test_fbx_in_blender(fbx_path)
            else:
                print(f"File not found: {fbx_path}")
        except (ValueError, IndexError):
            print("Invalid choice")
    else:
        # Test specific file
        fbx_path = Path(sys.argv[1])
        if fbx_path.exists():
            test_fbx_in_blender(fbx_path)
        else:
            print(f"File not found: {fbx_path}")

if __name__ == "__main__":
    main()