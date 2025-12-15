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
    const cliffBase = `${import.meta.env.BASE_URL}static/werewolf/cliff/`;
    const werewolfBase = `${import.meta.env.BASE_URL}static/werewolf/`;
    
    console.debug('[Town Loader] Loading cliff scene with splatmap...');

    const texLoader = new this.THREE.TextureLoader();
    
    // Load textures for cliff
    const splatMap = texLoader.load(cliffBase + 'cliff_splatmap.png');
    const texBase = texLoader.load(cliffBase + 'aerial_rocks_02_diff_1k.jpg');
    const texRed = texLoader.load(cliffBase + 'aerial_grass_rock_diff_1k.jpg');
    const texGreen = texLoader.load(cliffBase + 'rock_boulder_dry_diff_1k.jpg');

    // Configure detailed textures
    [texBase, texRed, texGreen].forEach(t => {
        t.wrapS = this.THREE.RepeatWrapping;
        t.wrapT = this.THREE.RepeatWrapping;
        t.encoding = this.THREE.sRGBEncoding;
        t.flipY = false; // Important for GLTF models
    });
    
    splatMap.encoding = this.THREE.LinearEncoding;
    splatMap.colorSpace = THREE.NoColorSpace;
    splatMap.flipY = false; // Match UV orientation

    // Custom material with splatmap logic
    const customMaterial = new this.THREE.MeshStandardMaterial({
        roughness: 1.0,
        metalness: 0.0,
        side: this.THREE.DoubleSide,
        color: 0xffffff, // Base color white so textures show up correctly
        map: texBase // Triggers USE_MAP and USE_UV defines
    });

    customMaterial.onBeforeCompile = (shader) => {
        shader.uniforms.splatMap = { value: splatMap };
        shader.uniforms.texBase = { value: texBase };
        shader.uniforms.texRed = { value: texRed };
        shader.uniforms.texGreen = { value: texGreen };
        shader.uniforms.repeat = { value: 10.0 };

        shader.fragmentShader = `
            uniform sampler2D splatMap;
            uniform sampler2D texBase;
            uniform sampler2D texRed;
            uniform sampler2D texGreen;
            uniform float repeat;

            vec4 hash4( vec2 p ) {
                return fract(sin(vec4( 1.0+dot(p,vec2(37.0,17.0)), 
                                       2.0+dot(p,vec2(11.0,47.0)),
                                       3.0+dot(p,vec2(41.0,29.0)),
                                       4.0+dot(p,vec2(23.0,31.0))))*103.0);
            }

            vec4 textureNoTile( sampler2D samp, in vec2 uv )
            {
                vec2 p = floor( uv );
                vec2 f = fract( uv );
                
                // derivatives (for correct mipmapping)
                vec2 ddx = dFdx( uv );
                vec2 ddy = dFdy( uv );
                
                // voronoi contribution
                vec4 va = vec4( 0.0 );
                float wt = 0.0;
                for( int j=-1; j<=1; j++ )
                for( int i=-1; i<=1; i++ )
                {
                    vec2 g = vec2( float(i), float(j) );
                    vec4 o = hash4( p + g );
                    vec2 r = g - f + o.xy;
                    float d = dot(r,r);
                    float w = exp(-5.0*d );
                    vec4 c = textureGrad( samp, uv + o.zw, ddx, ddy );
                    va += w*c;
                    wt += w;
                }
                
                return va/wt;
            }
        ` + shader.fragmentShader;

        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <map_fragment>',
            `
            // Sample splatmap using standard UVs
            vec4 splat = texture2D(splatMap, vUv);
            
            // Sample textures with repeat using stochastic sampling
            vec4 colBase = textureNoTile(texBase, vUv * repeat);
            colBase = mapTexelToLinear(colBase);

            vec4 colRed = textureNoTile(texRed, vUv * repeat);
            colRed = mapTexelToLinear(colRed);

            vec4 colGreen = textureNoTile(texGreen, vUv * repeat);
            colGreen = mapTexelToLinear(colGreen);

            // Mix logic:
            // Base (Black) -> texBase
            // Red Channel -> texRed (Moss)
            // Green Channel -> texGreen (Rock 2) - DISABLED
            
            // Clean up overlapping weights
            float wRed = splat.r; // max(0.0, splat.r - splat.g);
            // float wGreen = max(0.0, splat.g - splat.r);

            // Start with base
            vec4 texelColor = colBase;
            
            // Mix in Red (Moss)
            texelColor = mix(texelColor, colRed, wRed);
            
            // Mix in Green (Rock 2)
            // texelColor = mix(texelColor, colGreen, wGreen);

            diffuseColor *= texelColor;
            `
        );
    };

    const loadGLB = (url, isCliff) => {
        this.gltfLoader.load(
            url,
            (gltf) => {
                console.debug(`[Town Loader] Loaded ${url}`);
                const model = gltf.scene;
                model.scale.setScalar(14.0); 
                model.position.set(0, 0, 0);

                model.traverse((child) => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                        
                        if (isCliff) {
                            child.material = customMaterial;
                        } else {
                            // Ensure standard materials look okay
                            if (child.material && child.material.map) {
                                child.material.map.encoding = this.THREE.sRGBEncoding;
                            }
                        }
                    }
                });
                
                this.scene.add(model);
                if (isCliff) this.townModel = model;
            },
            undefined,
            (error) => {
                console.error(`[Town Loader] Error loading ${url}:`, error);
            }
        );
    };

    loadGLB(cliffBase + 'cliff_mesh.glb', true);
    loadGLB(werewolfBase + 'tower.glb', false);
    loadGLB(werewolfBase + 'town_center.glb', false);
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
