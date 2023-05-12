$(document).ready(function () {
  $("#submit").click(function () {
    let url = $("#url").val();
    console.log(url);
    $.ajax({
      type: "POST",
      url: "/download",
      contentType: "application/json",
      data: JSON.stringify({ url: url }),
      success: function (response) {
        console.log(response);
        // Handle success response from server
      },
      error: function (xhr, status, error) {
        console.log(xhr.responseText);
        // Handle error response from server
      },
    });
  });
});
