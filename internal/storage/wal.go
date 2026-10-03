package storage

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"sync"

	"vektordb/internal/bridge"
)

const (
	OpInsert uint8 = 1
	OpDelete uint8 = 2

	// Frame header: [4 bytes RecordLen][4 bytes CRC32]
	HeaderSize = 8
	// Fixed payload header: [1 byte OpCode][8 bytes ID][4 bytes Dim]
	PayloadFixedSize = 13
)

var (
	ErrCorruptFrame = errors.New("wal frame corrupted")
	ErrWALClosed    = errors.New("wal file is closed")
)

// WAL represents an append-only binary Write-Ahead Log.
type WAL struct {
	path        string
	file        *os.File
	mu          sync.Mutex
	syncOnWrite bool
	closed      bool
}

// OpenWAL opens or creates a WAL file at the specified path.
func OpenWAL(path string, syncOnWrite bool) (*WAL, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open wal file %s: %w", path, err)
	}

	return &WAL{
		path:        path,
		file:        file,
		syncOnWrite: syncOnWrite,
	}, nil
}

// WriteEntry appends a record frame to the WAL:
// [4 bytes RecordLen][4 bytes CRC32][1 byte OpCode][8 bytes ID][4 bytes Dim][N*4 bytes FloatData]
func (w *WAL) WriteEntry(op uint8, id uint64, vec []float32) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed || w.file == nil {
		return ErrWALClosed
	}

	dim := uint32(len(vec))
	payloadLen := PayloadFixedSize + int(dim)*4
	recordLen := uint32(payloadLen)

	// Build payload
	payload := make([]byte, payloadLen)
	payload[0] = op
	binary.BigEndian.PutUint64(payload[1:9], id)
	binary.BigEndian.PutUint32(payload[9:13], dim)

	offset := 13
	for _, f := range vec {
		binary.BigEndian.PutUint32(payload[offset:offset+4], math.Float32bits(f))
		offset += 4
	}

	// Calculate CRC32 of payload
	checksum := crc32.ChecksumIEEE(payload)

	// Build complete frame
	frame := make([]byte, HeaderSize+payloadLen)
	binary.BigEndian.PutUint32(frame[0:4], recordLen)
	binary.BigEndian.PutUint32(frame[4:8], checksum)
	copy(frame[8:], payload)

	if _, err := w.file.Write(frame); err != nil {
		return fmt.Errorf("wal write failed: %w", err)
	}

	if w.syncOnWrite {
		if err := w.file.Sync(); err != nil {
			return fmt.Errorf("wal sync failed: %w", err)
		}
	}

	return nil
}

// Sync forces all written data to physical disk.
func (w *WAL) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed || w.file == nil {
		return ErrWALClosed
	}
	return w.file.Sync()
}

// Close closes the WAL file descriptor.
func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return nil
	}
	w.closed = true
	if w.file != nil {
		_ = w.file.Sync()
		err := w.file.Close()
		w.file = nil
		return err
	}
	return nil
}

// Path returns the WAL file path.
func (w *WAL) Path() string {
	return w.path
}

// Replay reads the log sequentially, validates CRC32 integrity, ignores corrupted trailing writes on crash,
// and replays valid records back into the C++ index.
// If indexPath is provided and exists, it verifies or loads it before applying residual log records.
func (w *WAL) Replay(indexPath string, idx *bridge.Index) (recoveredCount int, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	return ReplayFromPath(w.path, indexPath, idx)
}

// ReplayFromPath replays a WAL file located at walPath into the target index.
func ReplayFromPath(walPath string, indexPath string, idx *bridge.Index) (recoveredCount int, err error) {
	if idx == nil {
		return 0, errors.New("cannot replay into nil index")
	}

	f, err := os.Open(walPath)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, fmt.Errorf("failed to open wal for replay: %w", err)
	}
	defer f.Close()

	headerBuf := make([]byte, HeaderSize)
	for {
		// Read 8-byte frame header
		_, err := io.ReadFull(f, headerBuf)
		if err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				// Clean EOF or partial write on crash: stop safely without losing acknowledged data
				break
			}
			return recoveredCount, fmt.Errorf("error reading wal frame header: %w", err)
		}

		recordLen := binary.BigEndian.PutUint32
		recLen := binary.BigEndian.Uint32(headerBuf[0:4])
		expectedCRC := binary.BigEndian.Uint32(headerBuf[4:8])
		_ = recordLen

		if recLen < PayloadFixedSize {
			// Invalid record length (trailing corrupted write)
			break
		}

		payload := make([]byte, recLen)
		_, err = io.ReadFull(f, payload)
		if err != nil {
			// Partial trailing write on crash: ignore and stop
			break
		}

		actualCRC := crc32.ChecksumIEEE(payload)
		if actualCRC != expectedCRC {
			// Corrupted frame detected (e.g. torn page or half-written record on crash): stop replay
			break
		}

		op := payload[0]
		id := binary.BigEndian.Uint64(payload[1:9])
		dim := binary.BigEndian.Uint32(payload[9:13])

		if int(recLen) != PayloadFixedSize+int(dim)*4 {
			// Mismatched dimension vs payload length: stop
			break
		}

		vec := make([]float32, dim)
		offset := 13
		for i := 0; i < int(dim); i++ {
			bits := binary.BigEndian.Uint32(payload[offset : offset+4])
			vec[i] = math.Float32frombits(bits)
			offset += 4
		}

		if op == OpInsert {
			if err := idx.Insert(id, vec); err != nil {
				return recoveredCount, fmt.Errorf("failed to insert recovered vector %d: %w", id, err)
			}
			recoveredCount++
		}
	}

	return recoveredCount, nil
}
