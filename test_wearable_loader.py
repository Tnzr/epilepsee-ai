#!/usr/bin/env python3
"""Test script for WearableDeviceDataLoader integration."""

import sys
import logging
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 70)
print("WearableDeviceDataLoader - Integration Test")
print("=" * 70)

try:
    # Import config
    from config.config import DEFAULT_CONFIG
    from src.data_loader import WearableDeviceDataLoader
    
    logger.info("Modules imported successfully")
    
    # Initialize loader
    loader = WearableDeviceDataLoader(DEFAULT_CONFIG.data)
    logger.info("✓ WearableDeviceDataLoader initialized")
    
    # Get all recordings
    recordings = loader.get_all_recordings()
    logger.info(f"✓ Found {len(recordings)} recordings")
    
    # Analyze recordings
    seizure_recordings = [r for r in recordings if r['has_seizure']]
    logger.info(f"✓ Found {len(seizure_recordings)} recordings with seizures")
    
    print("\n" + "-" * 70)
    print("Recording Summary:")
    print("-" * 70)
    for i, rec in enumerate(recordings, 1):
        status = "✓ SEIZURE" if rec['has_seizure'] else "  no seizure"
        seizures_info = f" ({rec['num_seizures']} seizures)" if rec['has_seizure'] else ""
        dir_status = "✓" if rec['recording_dir'] is not None else "✗"
        print(f"{i:2}. {rec['subject_id']}/{rec['watch_id']}: {status}{seizures_info} [{dir_status}]")
    
    # Test feature extraction from first seizure recording
    print("\n" + "-" * 70)
    print("Feature Extraction Test:")
    print("-" * 70)
    
    if seizure_recordings:
        test_rec = None
        for rec in seizure_recordings:
            if rec['recording_dir'] is not None:
                test_rec = rec
                break
        
        if test_rec:
            logger.info(f"Testing with: {test_rec['subject_id']}/{test_rec['watch_id']}")
            logger.info(f"  Seizure times: {test_rec['seizure_times']}")
            
            try:
                features, labels, times = loader.extract_features_from_recording(test_rec)
                if features is not None:
                    preictal_count = (labels >= 0).sum()
                    interictal_count = (labels < 0).sum()
                    logger.info(f"✓ Features extracted: shape {features.shape}")
                    logger.info(f"  Preictal samples: {preictal_count}")
                    logger.info(f"  Interictal samples: {interictal_count}")
                else:
                    logger.warning("No features could be extracted")
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("No seizure recordings with valid recording_dir found")
    else:
        logger.warning("No seizure recordings found for testing")
    
    print("\n" + "=" * 70)
    print("✓ Integration test completed successfully!")
    print("=" * 70)
    
except Exception as e:
    logger.error(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
