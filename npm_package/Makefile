.PHONY: all

all:
	./wasm-pack-build.sh browser
	./wasm-pack-build.sh nodejs
	cp package.json.in pkg/package.json

clean:
	rm -rf pkg
