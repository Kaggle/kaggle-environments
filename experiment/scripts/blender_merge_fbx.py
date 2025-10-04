#!/usr/bin/env python3
"""
Blender script for merging FBX animations.
This script should be run with Blender's Python interpreter.

Usage:
    blender --background --python blender_merge_fbx.py -- <rigged_fbx> <output_fbx> <anim1.fbx> <anim2.fbx> ...
"""

import bpy
import sys
import os
import json
from pathlib import Path

def clear_scene():
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear orphan data
    bpy.ops.outliner.orphans_purge(do_recursive=True)

def import_rigged_model(filepath, texture_path=None):
    """Import the rigged FBX model and apply texture if provided."""
    print(f"Importing rigged model: {filepath}")
    bpy.ops.import_scene.fbx(
        filepath=filepath,
        use_custom_normals=True,
        use_image_search=True,
        use_alpha_decals=False,
        decal_offset=0.0,
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
    
    # Apply texture if provided
    if texture_path and os.path.exists(texture_path):
        print(f"Loading texture: {texture_path}")
        # Load the texture
        texture_img = bpy.data.images.load(texture_path)
        
        # Apply texture to all mesh objects
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                # Ensure the object has materials
                if len(obj.data.materials) == 0:
                    # Create a new material
                    mat = bpy.data.materials.new(name="TexturedMaterial")
                    mat.use_nodes = True
                    obj.data.materials.append(mat)
                
                # Apply texture to all materials
                for mat in obj.data.materials:
                    if mat and mat.use_nodes:
                        nodes = mat.node_tree.nodes
                        
                        # Find or create an image texture node
                        tex_node = None
                        for node in nodes:
                            if node.type == 'TEX_IMAGE':
                                tex_node = node
                                break
                        
                        if not tex_node:
                            tex_node = nodes.new(type='ShaderNodeTexImage')
                            tex_node.location = (-300, 300)
                        
                        # Set the texture
                        tex_node.image = texture_img
                        
                        # Connect to the base color if using Principled BSDF
                        bsdf = nodes.get("Principled BSDF")
                        if bsdf:
                            links = mat.node_tree.links
                            links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
                        
                        print(f"  Applied texture to material: {mat.name}")
    
    # Find and return the armature
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None

def import_animation(filepath, armature, action_name):
    """Import an animation FBX and apply it as an action to the armature."""
    print(f"Importing animation: {action_name} from {filepath}")
    
    # Store current objects
    before_import = set(bpy.context.scene.objects)
    
    # Import the animation FBX
    bpy.ops.import_scene.fbx(
        filepath=filepath,
        use_custom_normals=True,
        use_image_search=True,
        use_alpha_decals=False,
        decal_offset=0.0,
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
    
    # Find newly imported objects
    after_import = set(bpy.context.scene.objects)
    new_objects = after_import - before_import
    
    # Find the imported armature
    imported_armature = None
    for obj in new_objects:
        if obj.type == 'ARMATURE':
            imported_armature = obj
            break
    
    if imported_armature and imported_armature.animation_data:
        # Get the action from the imported armature
        imported_action = imported_armature.animation_data.action
        if imported_action:
            # Rename the action
            imported_action.name = action_name
            
            # Store the action for later use
            if armature.animation_data is None:
                armature.animation_data_create()
            
            # Add to NLA tracks
            nla_track = armature.animation_data.nla_tracks.new()
            nla_track.name = action_name
            nla_strip = nla_track.strips.new(
                name=action_name,
                start=1,
                action=imported_action
            )
            
            print(f"  Added action '{action_name}' to NLA track")
    
    # Clean up imported objects (except the action data)
    for obj in new_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

def export_merged_fbx(filepath, embed_textures=True):
    """Export the merged FBX with all animations and optionally embedded textures."""
    print(f"Exporting merged FBX to: {filepath}")
    print(f"  Embed textures: {embed_textures}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Copy textures to the same directory as the FBX for relative paths
    if embed_textures:
        output_dir = os.path.dirname(filepath)
        for img in bpy.data.images:
            if img.filepath and not img.packed_file:
                # Copy texture to output directory
                texture_name = os.path.basename(img.filepath)
                texture_dest = os.path.join(output_dir, texture_name)
                if os.path.exists(img.filepath) and img.filepath != texture_dest:
                    import shutil
                    shutil.copy2(img.filepath, texture_dest)
                    print(f"  Copied texture: {texture_name}")
    
    # Find armature and make sure NLA tracks are not muted so all actions are exported
    armature = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
    
    if armature and armature.animation_data:
        # Unmute all NLA tracks to ensure actions are exported properly
        for track in armature.animation_data.nla_tracks:
            track.mute = False
    
    bpy.ops.export_scene.fbx(
        filepath=filepath,
        check_existing=False,
        filter_glob="*.fbx",
        use_selection=False,
        use_active_collection=False,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_NONE',
        use_space_transform=True,
        bake_space_transform=False,
        object_types={'ARMATURE', 'MESH', 'EMPTY'},
        use_mesh_modifiers=True,
        use_mesh_modifiers_render=True,
        mesh_smooth_type='OFF',
        use_subsurf=False,
        use_mesh_edges=False,
        use_tspace=False,
        use_custom_props=False,
        add_leaf_bones=True,
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        use_armature_deform_only=False,
        armature_nodetype='NULL',
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=True,
        bake_anim_use_all_actions=True,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=1.0,
        path_mode='COPY' if embed_textures else 'AUTO',
        embed_textures=embed_textures,
        batch_mode='OFF',
        use_batch_own_dir=True,
        axis_forward='-Z',
        axis_up='Y'
    )

def main():
    """Main function to process FBX merging."""
    # Get command line arguments after '--'
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        print("Error: No arguments provided")
        sys.exit(1)
    
    if len(argv) < 3:
        print("Error: Not enough arguments")
        print("Usage: blender --background --python blender_merge_fbx.py -- <rigged_fbx> <output_fbx> <animations_json> [texture_path]")
        sys.exit(1)
    
    rigged_fbx = argv[0]
    output_fbx = argv[1]
    animations_json = argv[2]
    texture_path = argv[3] if len(argv) > 3 else None
    
    # Load animations data
    with open(animations_json, 'r') as f:
        animations = json.load(f)
    
    print(f"Processing merge:")
    print(f"  Rigged model: {rigged_fbx}")
    print(f"  Output file: {output_fbx}")
    print(f"  Animations: {list(animations.keys())}")
    
    # Clear the scene
    clear_scene()
    
    # Import the rigged model with texture
    armature = import_rigged_model(rigged_fbx, texture_path)
    if not armature:
        print("Error: No armature found in rigged model")
        sys.exit(1)
    
    print(f"Found armature: {armature.name}")
    
    # Import each animation
    for action_name, anim_path in animations.items():
        if os.path.exists(anim_path):
            import_animation(anim_path, armature, action_name)
        else:
            print(f"  Warning: Animation file not found: {anim_path}")
    
    # Export the merged FBX with embedded textures
    export_merged_fbx(output_fbx, embed_textures=True)
    
    print("Merge completed successfully!")

if __name__ == "__main__":
    main()