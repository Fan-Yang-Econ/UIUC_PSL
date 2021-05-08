# Base on offical Node.js Alpine image
FROM node:alpine

# Set working directory
WORKDIR /usr/app

# First we copy over our package.json and package-lock.json files, then we install our dependencies, and then copy the rest of our files.
# The reason we do this is to take advantage of Docker’s caching system. At each step (or layer) in the Dockerfile process, Docker will cache the result so that it can re-build the image much faster next time.
# If we were to copy over all of the files in one go, changing any file in our project would cause Docker to install the dependencies again on subsequent builds, even if the dependencies have not changed.
# By copying the files over in two stages, Docker can use the cached layers, including the layer with the dependencies, if our package.json or package-lock.json files have not changed.

COPY package*.json ./

# Install dependencies
# With the --production flag (or when the NODE_ENV environment variable is set to production), npm will not install modules listed in devDependencies.
# NOTE: The --production flag has no particular meaning when adding a dependency to a project.
# RUN npm install --production

# Copy all files
# By copying the files over in two stages, Docker can use the cached layers, including the layer with the dependencies, if our package.json or package-lock.json files have not changed.
COPY . ./

# Build app
## ALREADY BUILD IN local or CodeBuild stage!
RUN npm run build

# Expose the listening port
EXPOSE 8080

# Run container as non-root (unprivileged) user
# The node user is provided in the Node.js Alpine base image

# By default, Docker will run containers as root. Running Node.js as root can lead to security issues. For this reason, after all setup and running our build, we then switch to a non-root, unprivileged user, node. This is provided for us by the base node:alpine image.
USER node

# Run npm start script when container starts
# PM2 is a production ready process manager for Node.js applications.
# We could simply run npm start directly when our container starts, as above, but this is a bad idea in production. If our app crashes for some reason, our Node.js process will exit and our app will no longer be available.
# PM2 solves this problem by ensuring that our app is always restarted after crashing.
CMD [ "./node_modules/pm2/bin/pm2-runtime", "npx", "--", "next", "start", "-p", "8080" ]


# ~~~ commands to test ~~~
# docker image build -t movie_rating_app:latest .
# docker container rm movie_app_container
# docker run -p 8000:8080 --name movie_app_container --link movie_backend_container:movieratingbackend movie_rating_app:latest


#  zip en_app_version_movie_rating Dockerrun.aws.json
#  aws s3 cp ./en_app_version_movie_rating.zip s3://warrens-data-pipeline/en_app_version_movie_rating --profile warrens_developer


# $(aws --profile warrens_developer ecr get-login --no-include-email --region us-east-1)

# docker tag movie_rating_app:latest 060259987605.dkr.ecr.us-east-1.amazonaws.com/movie_rating_app:latest
# docker tag movie_rating_backend:latest 060259987605.dkr.ecr.us-east-1.amazonaws.com/movie_rating_backend:latest

# docker push 060259987605.dkr.ecr.us-east-1.amazonaws.com/movie_rating_app:latest
# docker push 060259987605.dkr.ecr.us-east-1.amazonaws.com/movie_rating_backend:latest
