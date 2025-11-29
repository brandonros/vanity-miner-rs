```shell
container system start
container build -f Dockerfile -t cuda-13-0-rust-builder
container run --rm -it --memory 8G -v $(pwd):/mnt cuda-13-0-rust-builder
cd /mnt
./scripts/build.sh
container system stop
```
