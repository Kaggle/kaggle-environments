export class AssetManager {
  constructor(loadingManager, modules) {
    this.loadingManager = loadingManager;
    this.THREE = modules.THREE;
    
    // Initialize loaders
    this.textureLoader = new this.THREE.TextureLoader(this.loadingManager);
    this.fbxLoader = new modules.FBXLoader(this.loadingManager);
    this.gltfLoader = new modules.GLTFLoader(this.loadingManager);
    this.rgbeLoader = new modules.RGBELoader(this.loadingManager);

    // Cache to store Promises for in-flight or completed loads
    this.cache = new Map();
  }

  /**
   * Internal helper to handle caching logic
   */
  _load(loader, url, onLoad) {
    if (this.cache.has(url)) {
      return this.cache.get(url);
    }

    const promise = new Promise((resolve, reject) => {
      loader.load(
        url,
        (asset) => {
          if (onLoad) onLoad(asset);
          resolve(asset);
        },
        (xhr) => {
          // Progress can be handled here if needed
        },
        (error) => {
          console.error(`[AssetManager] Error loading ${url}:`, error);
          // We remove the failed promise from cache so it can be retried
          this.cache.delete(url); 
          reject(error);
        }
      );
    });

    this.cache.set(url, promise);
    return promise;
  }

  loadTexture(url) {
    return this._load(this.textureLoader, url);
  }

  loadFBX(url) {
    return this._load(this.fbxLoader, url);
  }

  loadGLTF(url) {
    return this._load(this.gltfLoader, url);
  }

  loadHDR(url) {
    return this._load(this.rgbeLoader, url);
  }
  
  /**
   * Helper to load a batch of textures concurrently
   * @param {Object} urlsMap - Key-Value pairs of { name: url }
   * @returns {Promise<Object>} - Promise resolving to { name: texture }
   */
  async loadTextures(urlsMap) {
      const keys = Object.keys(urlsMap);
      const promises = keys.map(key => this.loadTexture(urlsMap[key]));
      
      const textures = await Promise.all(promises);
      
      const result = {};
      keys.forEach((key, index) => {
          result[key] = textures[index];
      });
      return result;
  }

  getToonGradientMap() {
    if (this.toonGradientMap) return this.toonGradientMap;

    // Create a 3-tone gradient map
    const colors = new Uint8Array([0, 128, 255]); // 3 tones: black/shadow, mid, bright
    // Or maybe 0, 128, 255 for standard 3-step. Let's try 3 steps.
    
    // Using DataTexture
    const texture = new this.THREE.DataTexture(colors, 3, 1, this.THREE.RedFormat);
    texture.minFilter = this.THREE.NearestFilter;
    texture.magFilter = this.THREE.NearestFilter;
    texture.needsUpdate = true;

    this.toonGradientMap = texture;
    return texture;
  }
}
