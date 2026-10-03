.PHONY: all build-core build-go build-web test bench run clean dev-web

CORE_DIR = core
BUILD_DIR = $(CORE_DIR)/build
BIN_DIR = bin

all: build-core build-go build-web

build-core:
	@echo "==> Building C++ Vector Core Shared Library..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && $(MAKE)

build-go: build-core
	@echo "==> Building Go VektorDB Daemon..."
	mkdir -p $(BIN_DIR)
	go build -o $(BIN_DIR)/vektordb ./cmd/vektordb

build-web:
	@echo "==> Building React/TypeScript Observability Console..."
	cd web && npm install && npm run build

test: build-core
	@echo "==> Running Crash Recovery & Ingestion Tests..."
	go test -v ./test/crash_recovery_test.go

bench: build-core
	@echo "==> Running Distance Kernel & Recall Benchmarks..."
	go test -v -run BenchmarkSIMDVsScalar ./test/...
	go test -v -run BenchmarkRecallVsEfSearch ./test/...

run: build-core build-go
	@echo "==> Starting VektorDB Daemon..."
	./$(BIN_DIR)/vektordb -port 8080 -data-dir ./data -seed 1500

dev-web:
	@echo "==> Starting Vite Dev Server..."
	cd web && npm run dev

clean:
	@echo "==> Cleaning Build Artifacts..."
	rm -rf $(BUILD_DIR) $(BIN_DIR) web/dist data
