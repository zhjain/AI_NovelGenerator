# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('vectorstore', 'vectorstore')
    ],
    hiddenimports=['typing_extensions', 
                    'langchain-openai',
                    'langgraph', 
                    'openai', 
                    'chromadb',
                    'nltk',
                    'sentence_transformers',
                    'scikit-learn',
                    'langchain-community',
                    'pydantic',
                    'pydantic.deprecated.decorator',
                    *collect_submodules('chromadb'),
                    'chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2',
                    'chromadb.telemetry.product.posthog',
                    'chromadb.api.segment',
                    'chromadb.db.impl',
                    'chromadb.db.impl.sqlite',
                    'chromadb.migrations',
                    'chromadb.migrations.embeddings_queue'
                    
                ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI_NovelGenerator_V1.1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico']
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI_NovelGenerator_V1.1',
)
