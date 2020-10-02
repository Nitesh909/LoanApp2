// ----------navbar slider function
var flag = true;
var flag2 = true;
$("#crossButton").click(navbarSlider);
function navbarSlider() {
  if (flag === true) {
    $("#navButton").attr("name", "close");
    $(".smallNav").slideDown(700);
    flag = false;
  } else {
    $("#navButton").attr("name", "list");
    $(".smallNav").slideUp(700);
    flag = true;
  }
};

//---------------navbar sticky top for big screen-------------------    
if (window.innerWidth >= 769){
var waypoint = new Waypoint({
  element: document.getElementById("sec1"),
  handler: function (direction) {
    if (direction == "down") {
      $("#bigNav").css("display", "none").css("position", "fixed").slideDown(400);
    }
    else {
      $("#bigNav").css("position", "relative");
    };
  },
  offset: -60
});};
//---------------navbar sticky bottom for small screen-------------------
if (window.innerWidth < 769){
  circleNav();
  var waypoint = new Waypoint({
    element: document.getElementById("sec3"),
    handler: function (direction) {
      if (direction == "down") {
        $("#circleNav").show();
      }
      else {
        $("#circleNav").hide();
      };
    },
    offset: 360
  });};
  
  function circleNav(){
    if (flag2 === true) {
      $("#circleNav").css("background-color","#EF5350");
      $("#circleNav ul").slideUp(600);
      flag2 = false;
    } else {
      $("#circleNav").css("background-color","white");
      $("#circleNav ul").slideDown(600);
      flag2 = true;
    }
  };
 
//---------------animation on scrolling-------------------   


var waypoint = new Waypoint({
  element: document.getElementById("projectNote"),
  handler: function () {
    $("#projectNote").addClass("animate__animated animate__slideInLeft").css("--animate-duration"," 2s").css("opacity","1");
  },
  offset: 90
});
var waypoint = new Waypoint({
  element: document.getElementById("sec1-box2"),
  handler: function () {
    $("#sec1-box2").addClass("animate__animated animate__zoomIn").css("--animate-duration"," 2s").css("opacity","1");
  },
  offset: 500
});
var waypoint = new Waypoint({
  element: document.getElementById("sec2-box1"),
  handler: function () {
    $("#sec2-box1").addClass("animate__animated animate__zoomIn").css("--animate-duration"," 2s").css("opacity","1");
  },
  offset: 500
});
var waypoint = new Waypoint({
  element: document.getElementById("sec2-box2"),
  handler: function () {
    $("#sec2-box2").addClass("animate__animated animate__slideInRight").css("--animate-duration"," 2s").css("opacity","1");
  },
  offset: 500
});
var waypoint = new Waypoint({
  element: document.getElementById("sec3"),
  handler: function () {
    $("#sec3").addClass("animate__animated animate__swing").css("--animate-duration"," 2s").css("opacity","1");
  },
  offset: 500
});


var waypoint = new Waypoint({
  element: document.getElementById("aboutUsIn1"),
  handler: function () {
    $("#aboutUsIn1").addClass("animate__animated animate__bounceInLeft").css("--animate-duration"," 2.2s").css("opacity","1");
  },
  offset: 550
});
var waypoint = new Waypoint({
  element: document.getElementById("aboutUsIn2"),
  handler: function () {
    $("#aboutUsIn2").addClass("animate__animated animate__bounceInDown").css("--animate-duration"," 2.2s").css("opacity","1");
  },
  offset: 550
});
var waypoint = new Waypoint({
  element: document.getElementById("aboutUsIn3"),
  handler: function () {
    $("#aboutUsIn3").addClass("animate__animated animate__bounceInUp").css("--animate-duration"," 2.2s").css("opacity","1");
  },
  offset: 550
});
var waypoint = new Waypoint({
  element: document.getElementById("aboutUsIn4"),
  handler: function () {
    $("#aboutUsIn4").addClass("animate__animated animate__bounceInRight").css("--animate-duration"," 2.2s").css("opacity","1");
  },
  offset: 550
});


var waypoint = new Waypoint({
  element: document.getElementById("sec5-box1"),
  handler: function () {
    $("#sec5-box1").addClass("animate__animated animate__rollIn").css("--animate-duration"," 2s").css("opacity","1");
  },
  offset: 550
});

var waypoint = new Waypoint({
  element: document.getElementById("sec5-box2"),
  handler: function () {
    $("#sec5-box2").addClass("animate__animated animate__rotateInDownRight").css("--animate-duration"," 2s").css("opacity","1");
  },
  offset: 550
});

