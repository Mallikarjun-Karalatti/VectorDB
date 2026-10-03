package storage

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"vektordb/internal/bridge"
)

// SnapshotManager coordinates periodic index checkpoints.
type SnapshotManager struct {
	baseDir     string
	interval    time.Duration
	stopCh      chan struct{}
	wg          sync.WaitGroup
	mu          sync.Mutex
	isSnapshotting bool
}

// NewSnapshotManager creates a snapshot manager.
func NewSnapshotManager(baseDir string, interval time.Duration) (*SnapshotManager, error) {
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create snapshot directory: %w", err)
	}

	return &SnapshotManager{
		baseDir:  baseDir,
		interval: interval,
		stopCh:   make(chan struct{}),
	}, nil
}

// CreateSnapshot serializes the C++ index to a snapshot file atomically.
func CreateSnapshot(idx *bridge.Index, targetPath string) error {
	if idx == nil {
		return fmt.Errorf("cannot snapshot nil index")
	}

	dir := filepath.Dir(targetPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory for snapshot %s: %w", targetPath, err)
	}

	tmpPath := targetPath + ".tmp"
	if err := idx.Save(tmpPath); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("failed to save index to %s: %w", tmpPath, err)
	}

	if err := os.Rename(tmpPath, targetPath); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("failed to rename temp snapshot to %s: %w", targetPath, err)
	}

	return nil
}

// LoadSnapshot restores a C++ index from an existing snapshot file.
func LoadSnapshot(snapshotPath string, dim, maxElements, metric int) (*bridge.Index, error) {
	if _, err := os.Stat(snapshotPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("snapshot file does not exist: %s", snapshotPath)
	}

	idx, err := bridge.LoadIndex(snapshotPath, dim, maxElements, metric)
	if err != nil {
		return nil, fmt.Errorf("failed to load snapshot from %s: %w", snapshotPath, err)
	}

	return idx, nil
}
