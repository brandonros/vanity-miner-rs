```shell
container system start
container build -f Dockerfile -t cuda-12-9-rust-builder
container run --rm -it --memory 8G -v $(pwd):/mnt cuda-12-9-rust-builder
cd /mnt
./scripts/build.sh
container system stop
```
