/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Set for the published static build: no API process is behind it. */
  readonly VITE_STATIC?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
