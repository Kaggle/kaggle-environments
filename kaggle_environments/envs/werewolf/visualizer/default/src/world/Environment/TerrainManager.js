export class TerrainManager {
  constructor(scene, THREE, FBXLoader, GLTFLoader) {
    this.scene = scene;
    this.THREE = THREE;
    this.fbxLoader = new FBXLoader();
    this.gltfLoader = new GLTFLoader();

    this.islandModel = null;
    this.townModel = null;

    this.init();
  }

  init() {
    // this.loadIslandModel();
    this.loadTownModel();
    // this.loadGround();
  }

  loadIslandModel() {
    const textureLoader = new this.THREE.TextureLoader();

    const baseTexture = textureLoader.load(`${import.meta.env.BASE_URL}static/werewolf/island/_0930062431_texture.png`);
    const normalTexture = textureLoader.load(
      `${import.meta.env.BASE_URL}static/werewolf/island/_0930062431_texture_normal.png`
    );
    const metallicTexture = textureLoader.load(
      `${import.meta.env.BASE_URL}static/werewolf/island/_0930062431_texture_metallic.png`
    );
    const roughnessTexture = textureLoader.load(
      `${import.meta.env.BASE_URL}static/werewolf/island/_0930062431_texture_roughness.png`
    );

    [baseTexture, normalTexture, metallicTexture, roughnessTexture].forEach((texture) => {
      texture.encoding = this.THREE.sRGBEncoding;
      texture.flipY = true;
    });

    this.fbxLoader.load(
      `${import.meta.env.BASE_URL}static/werewolf/island/_0930062431_texture.fbx`,
      (fbx) => {
        fbx.scale.setScalar(0.02);
        fbx.position.y = -19.8;
        fbx.rotation.y = Math.PI / 8;

        fbx.traverse((child) => {
          if (child.isMesh) {
            const material = new this.THREE.MeshStandardMaterial({
              map: baseTexture,
              normalMap: normalTexture,
              normalScale: new this.THREE.Vector2(0.5, 0.5),
              metalnessMap: metallicTexture,
              roughnessMap: roughnessTexture,
              metalness: 0.1,
              roughness: 0.95,
              envMapIntensity: 0.2,
              color: new this.THREE.Color(0.75, 0.75, 0.75),
              side: this.THREE.DoubleSide,
            });

            child.material = material;
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });

        this.scene.add(fbx);
        this.islandModel = fbx;
        console.debug('Island model loaded successfully');
      },
      (progress) => {
        console.debug('Loading island model:', ((progress.loaded / progress.total) * 100).toFixed(2) + '%');
      },
      (error) => {
        console.error('Error loading island model:', error);
        // Fallback
        const groundGeometry = new this.THREE.CircleGeometry(20, 64);
        const groundMaterial = new this.THREE.MeshStandardMaterial({
          color: 0x1a1a2a,
          roughness: 1,
          metalness: 0,
          transparent: true,
          opacity: 0.95,
        });
        const ground = new this.THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -0.1;
        ground.receiveShadow = true;
        this.scene.add(ground);
      }
    );
  }

  loadTownModel() {
    const townModelPath = `${import.meta.env.BASE_URL}static/werewolf/town/cliff_grass_rock_blend.glb`;
    console.debug(`[Town Loader] Attempting to load model from: ${townModelPath}`);

    this.gltfLoader.load(
      townModelPath,
      (gltf) => {
        console.debug('[Town Loader] Model loaded successfully.');
        const model = gltf.scene;
        // Adjust scale/rotation as appropriate for the new model
        model.scale.setScalar(14.0);
        model.position.set(0, 0, 0);
        
        model.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            if (child.material) {
                // Reset material modifiers to ensure texture color shows purely
                if (child.material.map) {
                    child.material.map.encoding = this.THREE.sRGBEncoding;
                }
                child.material.color.setHex(0xffffff); // Ensure base color is white
                child.material.metalness = 0.0;
                child.material.roughness = 1.0;
                child.material.emissive.setHex(0x000000);
            }
          }
        });
        this.scene.add(model);
        this.townModel = model;
      },
      (progress) => {
        console.debug('[Town Loader] Loading progress: ' + ((progress.loaded / progress.total) * 100).toFixed(2) + '%');
      },
      (error) => {
        console.error('[Town Loader] An error happened while loading the town model:', error);
      }
    );
  }

  loadGround() {
    console.debug('[Ground Loader] Creating realistic rocky terrain...');
    const textureLoader = new this.THREE.TextureLoader();

    const colorTexture = textureLoader.load(
      `${import.meta.env.BASE_URL}static/werewolf/ground/rocky_terrain_02_diff_1k.jpg`
    );
    const displacementTexture = textureLoader.load(
      `${import.meta.env.BASE_URL}static/werewolf/ground/rocky_terrain_02_disp_1k.png`
    );
    // roughness and normal maps removed due to EXRLoader issues

    [colorTexture, displacementTexture].forEach((texture) => {
      texture.wrapS = this.THREE.RepeatWrapping;
      texture.wrapT = this.THREE.RepeatWrapping;
      texture.repeat.set(16, 16);
    });

    const groundGeometry = new this.THREE.CircleGeometry(200, 128);
    const groundMaterial = new this.THREE.MeshStandardMaterial({
      map: colorTexture,
      displacementMap: displacementTexture,
      displacementScale: 0.5,
      roughness: 0.8, // Default high roughness for rock
    });

    const ground = new this.THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.75;
    ground.receiveShadow = true;
    this.scene.add(ground);
    console.debug('[Ground Loader] Rocky terrain created and added to the scene.');
  }
}
