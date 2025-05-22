let time = 0;

function run() {
  console.log("runs", time, new Date());

  // Update time dynamically before next call
  time += 1000;

  // Schedule the next run
  setTimeout(run, time);
}

// Initial call
run();
