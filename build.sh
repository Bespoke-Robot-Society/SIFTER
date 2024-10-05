#!/bin/bash
set -e
set -x
set -v
branch=$(git rev-parse --abbrev-ref HEAD)
basedir=$(basename "$PWD")

# lowercase for docker image naming
imgname=${branch,,}--${basedir,,}


# kill all containers based on the same branch
oldversions=$(docker ps -q --filter ancestor=$imgname)

if [ -n "$oldversions" ]; then
    docker kill $oldversions
fi

# (re)build
docker build -t $imgname .

# (re)start
docker run -d -p $1 $imgname

# cleanup old images (not reliable)
#docker rmi $(docker images --filter "dangling=true" -q --no-trunc)
