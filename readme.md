# Getting started with db

This took way too long so i'm going to spill out some nerdage rn. 
When you want to create a db with memgraph, kill existing versions with:

`docker-compose down -v`

this will destroy the volumes of the existing docker containers

then run 

`docker-compose up -d`

This will run an initialization script to create a user with auth permissions as well as build a new docker container on a single network containing your db dependencies.

1) THEN go to http://localhost:3000 where memgraph labs will be served

2) Click on new connection this will take you to the interface where you will connect your `lab`
to your actual docker db you created.

3) The host will be `memgraph` the port will be `7687` you don't need a db name.

4) once there you can login with your username password `memgraphmikey`

### THis took way too long to figure out but hopefully it is the end of my tribulations

## SO NOW YOU PROBABLY NEED SOME DATA BABY

To load data into the db, run `python3 -m db.travel_dataset_into_db` script. This will turn most (with the exception of some missing lines) of your csv data into relationships in the db.

After that, you might want to do some cool ml shit. `db.entrich_nodes_with_encoder_data` takes each node and its corresponding relationship, and executes a bunch of tasks in your celery queue to spit out summaries. These summaries are tokenized into embedding profiles that represent user behavior over multiple trips. The idea is to find users with similar travel habits and recommend the trips they have taken to each other.
