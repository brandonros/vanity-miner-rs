
```shell
container system start
container build -f Dockerfile.spirv -t spirv-rust-builder
container run --rm -it --memory 8G -v $(pwd):/mnt spirv-rust-builder
cd /mnt
./scripts/build.sh
container system stop
```
