// 1. Import THREE core
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import { SkeletonUtils } from 'three/examples/jsm/utils/SkeletonUtils.js';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { FilmPass } from 'three/examples/jsm/postprocessing/FilmPass.js';
import { Sky } from 'three/examples/jsm/objects/Sky.js';
import VolumetricFire from '../world/vendor/volumetric_fire/VolumetricFire.js';

//  Handle the "Global" requirement for legacy scripts
if (!window.THREE) {
  window.THREE = THREE;
}

//  Export everything as a standard object
export const ThreeModules = {
  THREE, // Export core THREE if you need it elsewhere
  OrbitControls,
  FBXLoader,
  SkeletonUtils,
  EXRLoader,
  CSS2DRenderer,
  CSS2DObject,
  EffectComposer,
  RenderPass,
  UnrealBloomPass,
  ShaderPass,
  FilmPass,
  Sky,
  VolumetricFire,
};
