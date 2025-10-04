#!/usr/bin/env python3
"""
Check the content of merged FBX files using Blender in background mode.
"""

import subprocess
import sys
from pathlib import Path

def check_fbx_content(model_type):
    """Check FBX content using Blender in background mode."""
    blender_path = '/Applications/Blender.app/Contents/MacOS/Blender'
    fbx_path = f'experiment/static/werewolf/models/{model_type}/merged/{model_type}_master.fbx'
    
    # Create a diagnostic script for Blender
    diagnostic_script = f"""
import bpy
import sys

# Clear the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Clear all actions to start fresh
for action in bpy.data.actions:
    bpy.data.actions.remove(action)

# Import the FBX
filepath = r'{fbx_path}'
print(f"\\n{'='*60}")
print(f"Checking: {model_type.upper()} model")
print(f"File: {{filepath}}")
print(f"{'='*60}")

bpy.ops.import_scene.fbx(
    filepath=filepath,
    use_custom_normals=True,
    use_image_search=True,
    use_anim=True,
    anim_offset=1.0,
    use_custom_props=True,
    use_custom_props_enum_as_string=True,
    ignore_leaf_bones=False,
    force_connect_children=False,
    automatic_bone_orientation=False,
    primary_bone_axis='Y',
    secondary_bone_axis='X',
    use_prepost_rot=True
)

# Find armature
armature = None
meshes = []
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        armature = obj
    elif obj.type == 'MESH':
        meshes.append(obj)

print(f"\\n📦 OBJECTS:")
print(f"  - Armature: {{armature.name if armature else 'NOT FOUND'}}")
print(f"  - Meshes: {{len(meshes)}} found")
for mesh in meshes:
    print(f"    • {{mesh.name}}")

if armature:
    print(f"\\n🦴 ARMATURE INFO:")
    print(f"  - Name: {{armature.name}}")
    print(f"  - Bones: {{len(armature.data.bones)}}")
    
    if armature.animation_data:
        print(f"\\n🎬 ANIMATION DATA:")
        
        # Check active action
        if armature.animation_data.action:
            action = armature.animation_data.action
            print(f"  - Active Action: {{action.name}}")
            print(f"    • FCurves: {{len(action.fcurves)}}")
            print(f"    • Frame Range: {{action.frame_range}}")
        else:
            print(f"  - Active Action: None")
        
        # List all actions in the file
        print(f"\\n  📋 ALL ACTIONS ({{len(bpy.data.actions)}}):")
        if len(bpy.data.actions) > 0:
            for action in bpy.data.actions:
                print(f"    • {{action.name}}:")
                print(f"      - FCurves: {{len(action.fcurves)}}")
                print(f"      - Frame Range: {{action.frame_range}}")
                # Sample some fcurves
                if len(action.fcurves) > 0:
                    sample_curve = action.fcurves[0]
                    print(f"      - Sample FCurve: {{sample_curve.data_path}}")
        else:
            print(f"    ⚠️  No actions found in file!")
        
        # Check NLA tracks
        print(f"\\n  🎞️  NLA TRACKS ({{len(armature.animation_data.nla_tracks) if armature.animation_data.nla_tracks else 0}}):")
        if armature.animation_data.nla_tracks and len(armature.animation_data.nla_tracks) > 0:
            for track in armature.animation_data.nla_tracks:
                print(f"    • Track: {{track.name}}")
                for strip in track.strips:
                    print(f"      - Strip: {{strip.name}}")
                    if strip.action:
                        print(f"        Action: {{strip.action.name}}")
                        print(f"        Frames: {{strip.frame_start}} - {{strip.frame_end}}")
        else:
            print(f"    ⚠️  No NLA tracks found!")
    else:
        print(f"\\n⚠️  NO ANIMATION DATA on armature!")

# Check for textures/materials
print(f"\\n🎨 MATERIALS & TEXTURES:")
for mesh in meshes:
    if len(mesh.data.materials) > 0:
        print(f"  Mesh: {{mesh.name}}")
        for mat in mesh.data.materials:
            if mat:
                print(f"    • Material: {{mat.name}}")
                if mat.use_nodes:
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            print(f"      - Texture: {{node.image.name}}")
                            print(f"        Path: {{node.image.filepath}}")

print(f"\\n{'='*60}\\n")
"""
    
    # Save the diagnostic script
    script_path = Path('temp_diagnostic.py')
    script_path.write_text(diagnostic_script)
    
    try:
        # Run Blender in background mode
        cmd = [blender_path, '--background', '--python', str(script_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Extract relevant output
        for line in result.stdout.split('\n'):
            if not line.startswith('Blender') and not line.startswith('Read prefs') and line.strip():
                print(line)
                
    finally:
        # Clean up
        if script_path.exists():
            script_path.unlink()

def main():
    """Check all merged FBX files."""
    models = ['doctor', 'seer', 'villager', 'werewolf']
    
    for model in models:
        check_fbx_content(model)

if __name__ == "__main__":
    main()