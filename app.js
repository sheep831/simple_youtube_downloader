const express = require("express");
const { spawn } = require("child_process");
const app = express();
const port = 3000;

//parse JSON data submitted using HTTP POST request
app.use(express.json());

app.use(express.static("public"));

app.get("/", (req, res) => {
  var dataToSend;
  // spawn new child process to call the python script
  const python = spawn("python", [
    "./python/youtube_downloader.py",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  ]);

  // collect data from script
  python.stdout.on("data", function (data) {
    console.log("Pipe data from python script ...");
    dataToSend = data.toString();
  });

  python.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  // in close event we are sure that stream from child process is closed
  python.on("close", (code) => {
    console.log(`child process close all stdio with code ${code}`);

    dataToSend += "Today is Friday";
    // send data to browser
    res.send(dataToSend);
  });
});

app.post("/download", (req, res) => {
  // get the YouTube video URL from the user
  console.log(req.body);
  const url = req.body.url;

  // spawn a new child process to call the youtube-dl Python script
  const python = spawn("python", ["./python/youtube_downloader.py", url]);

  python.stdout.on("data", (data) => {
    console.log(`stdout: ${data}`);
  });

  python.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  python.on("close", (code) => {
    console.log(`child process close all stdio with code ${code}`);
    res.send("Video downloaded successfully!");
  });
});

app.listen(port, () => console.log(`Example app listening on port ${port}`));
