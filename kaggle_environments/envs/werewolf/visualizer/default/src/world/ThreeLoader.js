export async function loadThree() {
  if (window.THREE) {
    return window.THREE;
  }

  const THREEModule = await import('https://cdn.jsdelivr.net/npm/three@0.118/build/three.module.js');
  const THREE = THREEModule.default || THREEModule;
  window.THREE = THREE; // Make global as some modules might expect it
  return THREE;
}

export async function loadThreeModules() {
  // 1. Ensure THREE is loaded and assigned to window.THREE *before* anything else.
  await loadThree();

  // 2. Now import dependent modules sequentially or in parallel, 
  // but explicitly separated from VolumetricFire if it's finicky, 
  // though Promise.all should be fine now that window.THREE is set.
  // Actually, let's keep VolumetricFire separate just to be safe.

  const [
    { OrbitControls },
    { FBXLoader },
    { SkeletonUtils },
    { EXRLoader },
    { CSS2DRenderer, CSS2DObject },
    { EffectComposer },
    { RenderPass },
    { UnrealBloomPass },
    { ShaderPass },
    { FilmPass },
    { Sky }
  ] = await Promise.all([
    import('https://cdn.jsdelivr.net/npm/three@0.118/examples/jsm/controls/OrbitControls.js'),
    import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/loaders/FBXLoader.js'),
    import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/utils/SkeletonUtils.js'),
    import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/loaders/EXRLoader.js'),
    import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/renderers/CSS2DRenderer.js'),
    import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/EffectComposer.js'),
    import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/RenderPass.js'),
    import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/UnrealBloomPass.js'),
    import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/ShaderPass.js'),
    import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/FilmPass.js'),
    import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/objects/Sky.js')
  ]);

  // 3. Load VolumetricFire *after* everything else is confirmed ready.
  // Using dynamic import with a relative path.
  const VolumetricFireModule = await import('../../static/volumetric_fire/VolumetricFire.js');
  const VolumetricFire = VolumetricFireModule.default || window.VolumetricFire;

  return {
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
    VolumetricFire
  };
}