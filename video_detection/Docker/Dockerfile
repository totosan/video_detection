FROM ultralytics/ultralytics:latest-jetson-jetpack4
# set working directory before copying
WORKDIR /video_detection

# copy all application files into the working directory
COPY . .

# verify files are copied
RUN ls -alh
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
# add entrypoint running python3 app.py
ENTRYPOINT ["python3", "app.py"]