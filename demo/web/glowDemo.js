// glowDemo.js
// Drives the Glow paper interactive demo.

const DELTA_TIME = 1.0 / 45.0; // Default, 45 FPS.
const SERVER = "http://0.0.0.0:5050" // Change to wherver you have the server running
const S3_PREFIX =
  "https://d4mucfpksywv.cloudfront.net/research-covers/glow/";
const IMAGE_SRC_PREFIX =
  "https://d4mucfpksywv.cloudfront.net/research-covers/glow/demo/media";
// const IMAGE_PLACEHOLDER_SRC = `${IMAGE_SRC_PREFIX}/placeholder3.png`;
const IMAGE_PLACEHOLDER_SRC = `${IMAGE_SRC_PREFIX}/placeholder4.png`;
const LOADING_IMAGE_SRC = `${IMAGE_SRC_PREFIX}/loading.png`;
const DOWNLOAD_IMAGE_SRC = `${IMAGE_SRC_PREFIX}/DownloadIcon.png`;
const MAX_IMAGE_SIZE = 1024; // dimension in pixels.

function onDocumentReady() {
  let container = $(".GlowDemo_Container");
  if (container !== undefined) {
    let tabControl = new TabControl(container, ["Manipulate", "Mix"]);

    let manipulateContainer = tabControl.GetDiv(0);
    let manipulateDemo = new FaceSlidersDemo(manipulateContainer, {
        "Smiling":            31,
        "Age":                39,
        "Narrow Eyes":        23,
        "Blonde Hair":        9,
        "Beard":              24
      },
      {
        startingImageIdx: 2,
        startingSliderAlphaIdsToAlphas: {
          31: 0.66,
          39: -0.66,
          23: 0.66,
          9: 0.66
        }
      }
    ); // Many more possible sliders exist between 0 and 40 on the server!
    window["GlowDemo_Manipulate"] = manipulateDemo;
    tabControl.UpdateWhileActive(0, manipulateDemo);
    tabControl.CallToShow(0, () => { manipulateDemo.Appear(); });
    tabControl.CallToHide(0, () => { manipulateDemo.Hide(); });

    let mixContainer = tabControl.GetDiv(1);
    let mixDemo = new FaceMixingDemo(mixContainer,
      {
        leftStartingImageIdx: 4,
        rightStartingImageIdx: 3
      }
    );
    window["GlowDemo_Mix"] = mixDemo;
    tabControl.UpdateWhileActive(1, mixDemo);
    tabControl.CallToShow(1, () => { mixDemo.Appear(); });
    tabControl.CallToHide(1, () => { mixDemo.Hide(); });

    window.setInterval(() => { tabControl.Update(); }, DELTA_TIME);
  }
  else {
    console.error("GlowDemo container not found.");
  }
}
$(onDocumentReady);

function downloadScreenshot(filename, divToScreenshot, options) {
  window.GlowDemoDownloadFileName = filename;
  html2canvas($(divToScreenshot).get(0)).then(
    function (canvas) {
      //console.log(canvas);

      document.body.appendChild(canvas);

      if (options && options.canvasCropRect) {
        // { x: Number, y: Number, width: Number, Height: Number } (in pixels.)
        window.GlowDemoCanvasCropRect = options.canvasCropRect;
      }
      else {
        window.GlowDemoCanvasCropRect = undefined;
      }

      Canvas2Image.saveAsPNG(canvas);

      document.body.removeChild(canvas);

      if (options && options.finishedCallback) {
        options.finishedCallback();
      }
    }
  );
}

// ============
// Demo Globals
// ============

// Latent-space vector cache: Image name + manipulation string -> [image, vector].
window.imageAndVectorCache = {};

const ServerOps = {

  // Align-Encode

  getCachedEncodingUrl: function (name) {
    return `${S3_PREFIX}${name}/align_encode.json`;
  },

  // Manipulations Cache

  getCachedManipUrl: function (name, alphaIds, alphas) {
    alphaStr = [];
    for (var i = 0; i < 40; i++) {
      alphaStr.push("0.0");
    }
    for (var i = 0; i < alphaIds.length; i++) {
      if (alphas[i] != 0) { // don't want to replace "0.0" with "0"
        alphaStr[alphaIds[i]] = String(alphas[i]);
      }
    }
    let url = S3_PREFIX + name + "/" + alphaStr.join("_") + ".png";
    return url;
  },

  getCachedMixUrl: function (name0, name1, mixValue) {
    return `${S3_PREFIX}${name0}_${name1}/${mixValue}.png`;
  },

  // General

  getJSON: function(url, successHandler, errorHandler) {
    $.ajax({
      url: url,
      type: "GET",
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: successHandler,
      error: errorHandler
    });
  },

  postAPI: function (apiPath, data, successHandler, errorHandler) {
    $.ajax({
      url: SERVER + apiPath,
      type: "POST",
      data: JSON.stringify(data),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: successHandler,
      error: errorHandler
    });
  }
}

// ===========
// Tab Control
// ===========

class TabControl {

  constructor(container, tabNames) {
    this.element = container;
    this.tabNames = tabNames;

    let tabLabelContainer = $(document.createElement("div"));
    tabLabelContainer.addClass("GlowDemo");
    tabLabelContainer.addClass("GlowDemo_TabLabelContainer");
    this.tabLabelContainer = tabLabelContainer;
    this.element.append(tabLabelContainer);

    this.tabLabels = [];
    let tabIdx = 0;
    for (let tabName of tabNames) {
      let tabLabel = $(document.createElement("div"));
      tabLabel.addClass("GlowDemo");
      tabLabel.addClass("GlowDemo_TabLabel");
      tabLabel.html(tabName);
      let curIndex = tabIdx;
      tabLabel.click(() => { this.onClick(curIndex); });
      this.tabLabels.push(tabLabel);
      this.tabLabelContainer.append(tabLabel);

      tabIdx += 1;
    }
    this.tabContainers = [];
    this.updateables = [];
    this.showCalls = [];
    this.hideCalls = [];
    for (let tabName of tabNames) {
      let tabContainer = $(document.createElement("div"));
      tabContainer.addClass("GlowDemo");
      tabContainer.addClass("GlowDemo_TabContent");
      this.tabContainers.push(tabContainer);
      this.element.append(tabContainer);

      this.updateables.push(null);
      this.showCalls.push(null);
      this.hideCalls.push(null);
    }

    this.activeTabIdx = 0;
    this.tabLabels[this.activeTabIdx].addClass("GlowDemo");
    this.tabLabels[this.activeTabIdx].addClass("GlowDemo_ActiveTab");
  }

  GetDiv(index) {
    if (this.tabContainers[index] === undefined) {
      console.error("Invalid tab index: " + index);
    }

    return this.tabContainers[index];
  }

  UpdateWhileActive(index, updateable) {
    if (this.updateables[index] === undefined) {
      console.error("Invalid tab index: " + index);
      return;
    }

    this.updateables[index] = updateable;
  }

  CallToShow(index, showCall) {
    if (this.showCalls[index] === undefined) {
      console.error("Invalid tab index: " + index);
      return;
    }

    this.showCalls[index] = showCall;

    if (index == this.activeTabIdx) {
      showCall();
    }
  }

  CallToHide(index, hideCall) {
    if (this.hideCalls[index] === undefined) {
      console.error("Invalid tab index: " + index);
      return;
    }

    this.hideCalls[index] = hideCall;

    if (index != this.activeTabIdx) {
      hideCall();
    }
  }

  Update() {
    let activeUpdateable = this.updateables[this.activeTabIdx];
    if (activeUpdateable !== undefined && activeUpdateable !== null) {
      if (activeUpdateable.Update) {
        activeUpdateable.Update();
      }
    }

    // Update tab box border radii.
    let numTabs = this.tabLabels.length;
    let idx = 0;
    let radius = 4;
    for (let tabLabel of this.tabLabels) {
      if (idx == 0) {
        tabLabel.css("border-radius", `${radius}px 0px 0px ${radius}px`);
      }
      else if (idx == numTabs - 1) {
        tabLabel.css("border-radius", `0px ${radius}px ${radius}px 0px`);
      }
      else {
        tabLabel.css("border-radius", `0px`);
      }

      idx += 1;
    }
  }

  onClick(tabIdx) {
    this.setActiveTab(tabIdx);
  }

  setActiveTab(tabIdx) {
    this.tabLabels[this.activeTabIdx].removeClass("GlowDemo_ActiveTab");
    let hideCall = this.hideCalls[this.activeTabIdx];
    if (hideCall) {
      hideCall();
    }

    this.activeTabIdx = tabIdx;

    this.tabLabels[this.activeTabIdx].addClass("GlowDemo_ActiveTab");
    let showCall = this.showCalls[this.activeTabIdx];
    if (showCall) {
      showCall();
    }
  }

}

// ====================
// Glow Demo Base Class
// ====================

class GlowDemoBase {

  constructor() {
    if (window.imageDataAndVectorCache === undefined) {
      window.imageDataAndVectorCache = {};
    }
    if (window.imageCacheEntries === undefined) {
      window.imageCacheEntries = {};
    }
    if (window.mixingCache === undefined) {
      window.mixingCache = {};
    }
    if (window.imageMixingCacheEntries === undefined) {
      window.imageMixingCacheEntries = {};
    }
  }

  get imageDataAndVectorCache() {
    return window.imageDataAndVectorCache;
  }

  get imageCacheEntries() {
    return window.imageCacheEntries;
  }

  get mixingCache() {
    return window.mixingCache;
  }

  get imageMixingCacheEntries() {
    return window.imageMixingCacheEntries;
  }

  getUserSlotData(userSlotIdx) {
    //console.log("getUserSlotData NYI");
    return undefined;
  }

  requestEncodingFromImg(image, img, imageAndResponseCallback, options) {
    // Caching presets:
    // If the image is a known preset image, we get a cached version of its
    // align-encode result.
    if (image.presetName) {
      let zeroAlphas = [];
      let alphaIds = [];
      for (let manipName in this.faceManipulations) {
        zeroAlphas.push(0);
        alphaIds.push(this.faceManipulations[manipName]);
      }

      let cachedEncodingUrl = ServerOps.getCachedEncodingUrl(
        image.presetName,
        alphaIds,
        zeroAlphas
      );

      ServerOps.getJSON(
        cachedEncodingUrl,
        (response) => {
          imageAndResponseCallback(image, response);
        },
        (errResponse) => {
          console.error(errResponse);
        }
      );
    }
    else {
      // General pathway for getting align_encode data.
      let dataUrl = null;
      let post = ServerOps.postAPI;
      let postIt = (data) => {
        post(
          "/api/align_encode",
          { img: data },
          (response) => {
            imageAndResponseCallback(image, response);
          },
          (errResponse) => { console.error(errResponse); }
        );
      };
      if (options && options.alreadyDataUrl) {
        dataUrl = img;
        postIt(dataUrl);
      }
      else {
        toDataURL(img.src, (data) => {
          postIt(data);
        });
      }
    }
  }

  getImageEncodingCacheName(image) {
    return image.name + "_0_0_0_0_0";
  }

  clearCacheForImage(image) {
    //console.log("OK, clearing cache for " + image.name);

    let encodingCacheName = this.getImageEncodingCacheName(image);
    //console.log("clearing cache for name " + encodingCacheName);
    this.imageDataAndVectorCache[encodingCacheName] = undefined;

    ensureListExistsAtKey(this.imageCacheEntries, image.name);
    //console.log(this.imageCacheEntries[image.name]);
    for (let cachedImageString of this.imageCacheEntries[image.name]) {
      this.imageDataAndVectorCache[cachedImageString] = undefined;
      //console.log("set to undefined: " + cachedImageString);
    }
    this.imageCacheEntries[image.name] = undefined;

    ensureListExistsAtKey(this.imageMixingCacheEntries, image.name);
    //console.log("clearing mixing cache for " + image.name);
    //console.log(this.imageMixingCacheEntries[image.name]);
    for (let cachedImageString of this.imageMixingCacheEntries[image.name]) {
      this.mixingCache[cachedImageString] = undefined;
      //console.log("setting " + cachedImageString + " to undefined");
    }
    this.imageMixingCacheEntries[image.name] = undefined;
  }

}

// ================
// Face Mixing Demo
// ================

class FaceMixingDemo extends GlowDemoBase {

  constructor(container, options) {
    super();

    this.container = container;

    let element = $(document.createElement("div"));
    element.addClass("GlowDemo");
    element.addClass("GlowDemo_FaceMixingDemo");
    this.container.append(element);
    this.element = element;

    // Input image container.
    let inputImagesContainer = $(document.createElement("div"));
    inputImagesContainer.addClass("GlowDemo");
    inputImagesContainer.addClass("GlowDemo_MixingInputImagesContainer");
    this.element.append(inputImagesContainer);
    this.inputImagesContainer = inputImagesContainer;

    // Left image selector.
    let selectorFrame0 = $(document.createElement("div"));
    selectorFrame0.addClass("GlowDemo");
    selectorFrame0.addClass("GlowDemo_MixingSelectorFrame");
    selectorFrame0.addClass("GlowDemo_MixingSelectorFrameLeft");
    this.inputImagesContainer.append(selectorFrame0);
    let imageSelector0 = new ImageSelector(selectorFrame0,
      (image) => {
        this.loadImage(image, imageSelector0, () => {
          this.onImageLoaded(image, imageSelector0);
        });
      },
      (imageRequestingUserUpload) => {
        this.requestUserFileForEncoding(imageRequestingUserUpload,
          this.selector0, () => {
            this.onImageLoaded(imageRequestingUserUpload, this.selector0);
          }
        );
      },
      {
        "labelName": "Left Input",
        "labelClass": "GlowDemo_LeftInputLabel"
      }
    );
    this.selector0 = imageSelector0;

    // Right image selector.
    let selectorFrame1 = $(document.createElement("div"));
    selectorFrame1.addClass("GlowDemo");
    selectorFrame1.addClass("GlowDemo_MixingSelectorFrame");
    selectorFrame1.addClass("GlowDemo_MixingSelectorFrameRight");
    this.inputImagesContainer.append(selectorFrame1);
    let imageSelector1 = new ImageSelector(selectorFrame1,
      (image) => {
        this.loadImage(image, imageSelector1, () => {
          this.onImageLoaded(image, imageSelector1);
        });
      },
      (imageRequestingUserUpload) => {
        this.requestUserFileForEncoding(imageRequestingUserUpload,
          this.selector1, () => {
            this.onImageLoaded(imageRequestingUserUpload, this.selector1);
          }
        );
      },
      {
        "labelName": "Right Input",
        "labelClass": "GlowDemo_RightInputLabel"
      }
    );
    this.selector1 = imageSelector1;

    // Output image and mixing slider container.
    let outputAndMixingSliderContainer = $(document.createElement("div"));
    outputAndMixingSliderContainer.addClass("GlowDemo");
    outputAndMixingSliderContainer.addClass(
      "GlowDemo_OutputAndMixingSliderContainer"
    );
    this.element.append(outputAndMixingSliderContainer);
    this.outputAndMixingSliderContainer = outputAndMixingSliderContainer;

    // Output image frame.
    let outputFrame = $(document.createElement("div"));
    outputFrame.addClass("GlowDemo");
    outputFrame.addClass("GlowDemo_MixingOutputFrame");
    this.outputAndMixingSliderContainer.append(outputFrame);
    this.outputFrame = outputFrame;
    let imageOutput = new ImageOutput(outputFrame, {
      startVisible: false,
      labelName: "Output",
      labelClass: "GlowDemo_MixingOutputLabel",
      preDownloadClickCallback: () => {
        this.mixingSlider.Disable();
        this.mixingSlider.ForceHide();
      },
      downloadClickCallback: () => {
        let download = () => {
          downloadScreenshot(
            `Glow - ${this.imageOutput.imageName}.png`,
            this.container,
            {
              finishedCallback: () => {
                this.mixingSlider.Enable();
                this.mixingSlider.UnForceHide();
              },
              canvasCropRect: {
                x: 0, y: 0, width: "auto", height: (x) => { return x - 80; }
              }
            }
          )
        };

        // Make sure the image selector is closed.
        if (this.selector0.isOpen || this.selector1.isOpen) {
          this.selector0.Close();
          this.selector1.Close();

          window.setTimeout(() => { download(); }, 200);
        }
        else {
          download();
        }
      }
    });
    this.imageOutput = imageOutput;

    this.leftInputImage = null;
    this.rightInputImage = null;
    this._currentMixingOutputValid = false;

    // Mixing hint.
    let mixingHint = new Hint(
      outputFrame,
      "Slide to mix. Touch either input to change.",
      {
        "extraClasses": ["GlowDemo_MixingHint"]
      }
    );
    this.mixingHint = mixingHint;
    this._seenFirstMix = false;
    this.onFirstMix = () => {
      this.mixingHint.Hide();
    }

    let sliderFrame = $(document.createElement("div"));
    sliderFrame.addClass("GlowDemo_SliderFrame");
    sliderFrame.addClass("GlowDemo");
    sliderFrame.addClass("GlowDemo_MixingSliderFrame");
    this.outputAndMixingSliderContainer.append(sliderFrame);
    this.sliderFrame = sliderFrame;

    let mixingSlider = new FaceSlider(
      sliderFrame,
      "Mix",
      9,
      false,
      {
        extraLabelClasses: ["GlowDemo_MixingSliderLabel"],
        extraSliderClasses: ["GlowDemo_MixingSlider"],
        placeLabelAfter: true,
        extraHiderClasses: ["GlowDemo_MixingSliderHider"],
        sliderAttributes: {
          "type": "range",
          "min": "0",
          "max": "1",
          "value": "0.5",
          "step": "0.25"
        }
      }
    );
    this.sliderFrame.append(mixingSlider);
    this.mixingSlider = mixingSlider;

    // Custom initialization state.
    if (options && options.leftStartingImageIdx) {
      this.selector0.onImageClicked(
        this.selector0.images[options.leftStartingImageIdx]
      );
    }
    if (options && options.rightStartingImageIdx) {
      this.selector1.onImageClicked(
        this.selector1.images[options.rightStartingImageIdx]
      );
    }
    if (options && options.leftStartingImageIdx && options.rightStartingImageIdx) {
      // Having two starting images will start a mix immediately, but we don't
      // want this to cause hints to disappear until the _user_ starts a mix.
      this._suppressFirstMixEvent = true;
    }
  }

  Appear() {
    this.element.css("display", "table");
  }

  Hide() {
    this.element.css("display", "none");
  }

  Update() {
    if (this.updateOnce) {
      this.updateOnce();
      this.updateOnce = null;
    }

    this.selector0.Update();
    this.selector1.Update();
    this.mixingHint.Update();
    this.imageOutput.Update();
    this.mixingSlider.Update();

    this.doSliderInteractionUpdate();
  }

  // ------------------
  // Slider Interaction
  // ------------------

  doSliderInteractionUpdate() {
    let selector0 = this.selector0;
    let selector1 = this.selector1;

    let sliderHeld = this.mixingSlider.isHeld;
    if (sliderHeld && !this._wasSliderHeld) {
      // User just started touching a slider.

      this.imageOutput.StartLoadingVisual();

      this._wasSliderHeld = true;
    }
    if (!sliderHeld && this._wasSliderHeld) {
      // User stopped touching a slider.

      if (selector0.isClosed
          && this.isImageReadyForMixing(selector0.currentSelectedImage)
          && selector1.isClosed
          && this.isImageReadyForMixing(selector1.currentSelectedImage)) {

        let adjustedSliderState = this.calculateMixCacheString(
          selector0.currentSelectedImage,
          selector1.currentSelectedImage,
          this.mixingSlider.value
        );
        if (adjustedSliderState != this._currentSliderState) {
          // Slider state changed.

          this.mixingSlider.Disable();

          this.requestMix(
            selector0.currentSelectedImage,
            selector1.currentSelectedImage,
            this.mixingSlider.value
          );

          this._currentSliderState = adjustedSliderState;
        }
        else {
          this.imageOutput.StopLoadingVisual();
        }
      }
      else {
        console.error("Slider was moved but the selectors weren't ready.");

        this.imageOutput.StopLoadingVisual();
      }

      this._wasSliderHeld = false;
    }

    if (!(selector0.isClosed
          && this.isImageReadyForMixing(selector0.currentSelectedImage)
          && selector1.isClosed
          && this.isImageReadyForMixing(selector1.currentSelectedImage))) {
      this.mixingSlider.Disable();
    }
  }

  loadImage(image, imageSelector, onLoadedCallback) {
    let imageCacheName = this.getImageEncodingCacheName(image);
    //console.log("attempted to get cache for " + imageCacheName);

    let cachedImageDataAndVector = this.imageDataAndVectorCache[imageCacheName];
    if (cachedImageDataAndVector === undefined) {
      // Not present in cache, so let's load it.
      //console.log("image " + imageCacheName + " not present in cache.");
      imageSelector.Lock();
      imageSelector.StartLoadingVisual();

      let post = ServerOps.postAPI;

      if (image instanceof UserImageChoice) {
        let userSlotIdx = image.userSlotIdx;

        let userSlotData = this.getUserSlotData(userSlotIdx);
        if (userSlotData) {
          //console.log("Data exists in user slot " + userSlotIdx);
        }
        else {
          //console.log("No data in user slot, asking for it");
          this.requestUserFileForEncoding(image, imageSelector, onLoadedCallback);

          // We can't detect any change if the user doesn't change the input
          // file, so just unlock.
          imageSelector.StopLoadingVisual();
          imageSelector.Unlock();
        }
      }
      else {
        // Preset image, just request the encoding for it.
        this.requestEncodingFromImg(
          image,
          image.domElement,
          (image, response) => {
            this.onImageEncodingResponse(
              image, imageSelector, response, onLoadedCallback);
          }
        );
      }
    }
    else {
      //console.log("image " + imageCacheName + " already in cache.");

      onLoadedCallback(image, imageSelector);
    }
  }

  requestUserFileForEncoding(userImage, imageSelector, onLoadedCallback) {
    let image = userImage;
    image.RequestUserFile(
      (inputFile) => {
        var loadingImage = loadImage(
          inputFile,
          ((exifAdjustedImgAsCanvas) => {
            loadImage.scale
            this.requestEncodingFromImg(
              image,
              exifAdjustedImgAsCanvas.toDataURL("image/png"),
              (image, response) => {
                this.onImageEncodingResponse(
                  image, imageSelector, response, onLoadedCallback,
                  {
                    clearImageCacheOnSuccess: true
                  }
                );
              },
              { alreadyDataUrl: true }
            );
          }).bind(this),
          {
            orientation: true,
            maxWidth: MAX_IMAGE_SIZE,
            maxHeight: MAX_IMAGE_SIZE,
            contain: true,
            crossOrigin: "anonymous"
          }
        );
        if (loadingImage) {
          imageSelector.StartLoadingVisual();
          imageSelector.Lock();
        }
        else {
          console.error("Error loading image file.");
        }
      },
      (errResponse) => {
        console.error(errResponse);
      }
    );
  }

  onImageEncodingResponse(image, imageSelector, response, onLoadedCallback,
                          options) {
    if (!response["face_found"]) {
      // No face in this image.
      imageSelector.Unlock();
      imageSelector.StopLoadingVisual();
      imageSelector.ShowNoFaceFoundPopup();
    }
    else {
      if (options && options.clearImageCacheOnSuccess) {
        this.clearCacheForImage(image);
      }

      let imageData = response.img[0];
      let latentVector = response.z[0];

      // Now that this input is loaded, modify the input image to reflect
      // the loaded input.
      image.SetImageSource(imageData);

      let imageCacheName = this.getImageEncodingCacheName(image);
      // Cache the latent vector associated with this input image.
      this.imageDataAndVectorCache[imageCacheName] = [imageData, latentVector];

      // Record cache operation for invalidation later.
      ensureListExistsAtKey(this.imageCacheEntries, image.name);
      this.imageCacheEntries[image.name].push(imageCacheName);
      //console.log("added to cache: " + imageCacheName);

      onLoadedCallback(image, imageSelector);
    }
  }

  onImageLoaded(image, imageSelector) {
    //console.log(image.name + " was loaded.");

    imageSelector.Unlock();
    imageSelector.StopLoadingVisual();

    if (imageSelector.currentSelectedImage === image) {
      let thisSelector = imageSelector;

      let otherSelector = null;
      if (thisSelector === this.selector0) {
        otherSelector = this.selector1;
      }
      else if (thisSelector === this.selector1) {
        otherSelector = this.selector0;
      }
      else {
        //console.error("Got onImageLoaded for an unexpected image selector.");
        return;
      }

      if (thisSelector.isClosed
          && this.isImageReadyForMixing(thisSelector.currentSelectedImage)
          && otherSelector.isClosed
          && this.isImageReadyForMixing(otherSelector.currentSelectedImage)) {

        this.requestMix(
          thisSelector.currentSelectedImage,
          otherSelector.currentSelectedImage
        );
      }
    }
    else {
      //console.log("However, this selector has "
      //   + imageSelector.currentSelectedImage.name + " open instead.");
    }
  }

  isImageReadyForMixing(image) {
    return this.imageDataAndVectorCache[this.getImageEncodingCacheName(image)] !==
      undefined;
  }

  requestMix(image0, image1, mixValue) {
    let latentVector0 =
      this.imageDataAndVectorCache[this.getImageEncodingCacheName(image0)][1];
    let latentVector1 =
      this.imageDataAndVectorCache[this.getImageEncodingCacheName(image1)][1];
    if (latentVector0 === undefined || latentVector1 === undefined) {
      console.error("Left and right images not loaded for mixing. "
        + "(No latent vectors.)");
      return;
    }

    if (mixValue === undefined) {
      mixValue = 0.5;
    }

    if (!this._seenFirstMix) {
      if (!this._suppressFirstMixEvent) {
        this.onFirstMix();

        this._seenFirstMix = true;
      }
      else {
        this._suppressFirstMixEvent = false;
      }
    }

    // Shortcut: If the mixValue is 0 or 1, just set the source to that full
    // image.
    if (mixValue == 0) {
      this.showAndCacheOutput(image0.domElement.src, image0, image1, mixValue,
        undefined);
      return;
    }
    else if (mixValue == 1) {
      this.showAndCacheOutput(image1.domElement.src, image0, image1, mixValue,
        undefined);
      return;
    }

    let mixCacheString = this.calculateMixCacheString(
      image0, image1, mixValue
    );
    let cachedMixData = this.mixingCache[mixCacheString];
    if (cachedMixData !== undefined) {
      //console.log("Mix cache already exists for " + mixCacheString);

      this.imageOutput.SetImageSource(cachedMixData,
        image0.name + " and " + image1.name);
      this.imageOutput.StopLoadingVisual();
      this.imageOutput.Appear();
      this.mixingSlider.value = mixValue;
      this.mixingSlider.Appear();
      this.mixingSlider.Enable();
    }
    else {
      //console.log("No cache for " + mixCacheString);

      this.imageOutput.StartLoadingVisual();

      let bothPresets = (!(image0 instanceof UserImageChoice)
        && !(image1 instanceof UserImageChoice));
      if (bothPresets) {
        // Shortcut! All the results for the presets have been precomputed.
        let cachedMixUrl = ServerOps.getCachedMixUrl(
          image0.presetName, image1.presetName, mixValue
        );
        toDataURL(cachedMixUrl, (imageData) => {
          this.showAndCacheOutput(imageData, image0, image1, mixValue,
            mixCacheString);
        });
      }
      else {
        // Ask the server to compute the result of the mix.
        ServerOps.postAPI(
          "/api/mix",
          {
            z1: latentVector0,
            z2: latentVector1,
            alpha: parseFloat(mixValue)
          },
          (response) => {
            //console.log(response);

            let imageData = response.img[0];

            this.showAndCacheOutput(imageData, image0, image1, mixValue,
              mixCacheString);
          },
          (errResponse) => {
            console.error(errResponse);
          }
        );
      }
    }

    this._currentSliderState = mixCacheString;
  }

  showAndCacheOutput(imageData, image0, image1, mixValue, mixCacheString) {
    this.imageOutput.SetImageSource(imageData,
      image0.name + " and " + image1.name);
    this.imageOutput.StopLoadingVisual();
    this.imageOutput.Appear();
    this.mixingSlider.value = mixValue;
    this.mixingSlider.Appear();
    this.mixingSlider.Enable();

    if (mixCacheString) {
      this.mixingCache[mixCacheString] = imageData;

      ensureListExistsAtKey(this.imageMixingCacheEntries, image0.name);
      this.imageMixingCacheEntries[image0.name].push(mixCacheString);
      //console.log("added " + image0.name + "mixing cache: " + mixCacheString);

      ensureListExistsAtKey(this.imageMixingCacheEntries, image1.name);
      this.imageMixingCacheEntries[image1.name].push(mixCacheString);
      //console.log("added " + image1.name + "mixing cache: " + mixCacheString);
    }
  }

  calculateMixCacheString(image0, image1, mixValue) {
    let name0 = image0.name;
    let name1 = image1.name;

    // If the two images are the same, they won't "mix" with any difference.
    if (name0 == name1) {
      mixValue = 0.5;
    }

    // Always sort the two names. If a swap is necessary, reverse the mix value.
    if (name0 > name1) {
      [name1, name0] = [name0, name1];
      mixValue = 1 - mixValue;
    }

    return `${name0}_${name1}_${mixValue}`;
  }

}

// =================
// Face Sliders Demo
// =================


class FaceSlidersDemo extends GlowDemoBase {
  constructor(container, faceManipulations, options) {
    super();

    this.container = container;

    // Name (client/friendly) to ID (server API) key/value pairs.
    this.faceManipulations = faceManipulations;

    // Prepare DOM.
    let element = $(document.createElement("div"));
    element.addClass("GlowDemo");
    element.addClass("GlowDemo_FaceSlidersDemo");
    this.container.append(element);
    this.element = element;

    // Load a single container for image selector and image output frames.
    let selectorAndOutputFrame = $(document.createElement("div"));
    selectorAndOutputFrame.addClass("GlowDemo");
    selectorAndOutputFrame.addClass("GlowDemo_SelectorAndOutput");
    element.append(selectorAndOutputFrame);
    this.selectorAndOutputFrame = selectorAndOutputFrame;

    // Load image selector.
    let selectorFrame = $(document.createElement("div"));
    selectorFrame.addClass("GlowDemo");
    selectorFrame.addClass("GlowDemo_SelectorFrame");
    selectorAndOutputFrame.append(selectorFrame);
    this.imageSelector = new ImageSelector(
      selectorFrame,
      (image) => { this.onImageSelected(image); },
      (imageRequestingUserUpload) => {
        this.requestUserFileForEncoding(imageRequestingUserUpload);
      },
      {
        inputLabelText: "Input"
      }
    );

    // Load output frame DOM.
    let outputFrame = $(document.createElement("div"));
    outputFrame.addClass("GlowDemo");
    outputFrame.addClass("GlowDemo_OutputFrame");
    selectorAndOutputFrame.append(outputFrame);
    this.imageOutput = new ImageOutput(outputFrame,
      {
        startVisible: false,
        preDownloadClickCallback: () => {
          this.lockSliders();
        },
        downloadClickCallback: () => {
          let download = () => {
            downloadScreenshot(
              `Glow - ${this.imageOutput.imageName}.png`,
              this.selectorAndOutputFrame,
              {
                finishedCallback: () => {
                  this.unlockSliders();
                }
              }
            )
          };

          // Make sure the image selector is closed.
          if (this.imageSelector.isOpen) {
            this.imageSelector.Close();

            window.setTimeout(() => { download(); }, 200);
          }
          else {
            download();
          }
        }
      }
    );

    // Load slider frame below the input and output frames.
    let sliderFrame = $(document.createElement("div"));
    sliderFrame.addClass("GlowDemo");
    sliderFrame.addClass("GlowDemo_SliderFrame");
    element.append(sliderFrame);

    // Load sliders into the slider frame.
    this.faceSliders = [];
    this.idsToSliders = {};
    for (let manipulationName in this.faceManipulations) {
      let faceSlider = new FaceSlider(
        sliderFrame,
        manipulationName,
        this.faceManipulations[manipulationName],
        false
      );
      this.faceSliders.push(faceSlider);
      this.idsToSliders[this.faceManipulations[manipulationName]] = faceSlider;
    }

    // Initialize interaction state.
    this._wereSlidersHeld = false;
    this._areSlidersLocked = false;

    // Initialize input image state (beyonce).
    // Also once beyonce is loaded, move "Smiling" slider to max.
    if (options && options.startingImageIdx) {
      this.imageSelector.onImageClicked(
        this.imageSelector.images[options.startingImageIdx]
      );
    }
    if (options && options.startingSliderAlphaIdsToAlphas) {
      this.oneTimeImageLoadedCallback = () => {
        this.requestManipulation(options.startingSliderAlphaIdsToAlphas);
      }
    }

    // Bind a hint to the image selector, to close after the first image click.
    this.selectorHint = new Hint(
      selectorAndOutputFrame,
      "Tap to choose a face.",
      {
        extraClasses: ["GlowDemo_SelectorHint"],
        startHidden: true
      }
    );
    this.imageSelector.firstOpenCallback = () => {
      this.selectorHint.Hide();
    }
    this.downloadHint = new Hint(
      selectorAndOutputFrame,
      "Tap to download.",
      {
        extraClasses: ["GlowDemo_DownloadHint"],
        startHidden: true
      }
    );
    this.imageOutput.firstClickOrHover = () => {
      this.downloadHint.SetHideTimer(10);
    }

    this.onNextOutputShown = () => {
      this.selectorHint.Show();
      this.downloadHint.Show();
    }

    // Bind a hint to upload your own image after the selector hint disappears.
    this.uploadHint = new Hint(
      selectorAndOutputFrame,
      "Or upload your own!",
      {
        extraClasses: ["GlowDemo_SelectorHint"],
        startHidden: true
      }
    );
    this.selectorHint.onHiddenOnce = () => {
      this.uploadHint.Show();
      this.uploadHint.SetHideTimer(14);
    }
  }

  Update() {
    // Everything that animates gets regular updates.
    this.imageSelector.Update();
    this.selectorHint.Update();
    this.downloadHint.Update();
    this.uploadHint.Update();
    this.imageOutput.Update();
    for (let faceSlider of this.faceSliders) {
      faceSlider.Update();
    }

    this.doSliderInteractionUpdate();
  }

  Appear() {
    this.element.css("display", "table");
  }

  Hide() {
    this.element.css("display", "none");
  }

  // -----------
  // Interaction
  // -----------

  doSliderInteractionUpdate() {
    if (this.imageSelector.isOpen && !this.areSlidersLocked) {
      this.lockSliders();
    }

    let anySlidersHeld = false;
    for (let faceSlider of this.faceSliders) {
      if (faceSlider.isHeld) {
        anySlidersHeld = true;
        break;
      }
    }
    if (anySlidersHeld && !this._wereSlidersHeld) {
      // A slider just started moving.
      this.imageOutput.StartLoadingVisual();

      this._wereSlidersHeld = true;
    }
    if (!anySlidersHeld && this._wereSlidersHeld) {
      // The user just stopped manipulating a slider.

      // Only need to do anything special if the sliders actually changed.
      let adjustedSliderState = this.calculateSliderState();
      let adjustedAndCurrentEqual =
        areContentsEqual(adjustedSliderState, this._sliderState);
      if (adjustedAndCurrentEqual) {
        this.imageOutput.StopLoadingVisual();
      }
      else {
        this.lockSliders();

        // Make a request for the new slider state.
        this.requestManipulation(adjustedSliderState);

        this._sliderState = adjustedSliderState;
      }

      this._wereSlidersHeld = false;
    }

    if (anySlidersHeld) {
      this._noSliderMotionTimer = 0;
    }
  }

  // ---------------
  // Image Selection
  // ---------------

  onImageSelected(image) {
    if (this.currentInputImage != image
        || (image instanceof UserImageChoice
            && (this.getUserSlotData(image.userSlotIdx) === undefined))) {
      this.clearSliderState();

      this.currentInputImage = image;

      this.prepareImageManipulation();
    }
  }

  // -----------------
  // Slider Management
  // -----------------

  calculateSliderState() {
    let sliderState = {};
    for (let slider of this.faceSliders) {
      sliderState[parseFloat(slider.faceManipulationId)] = slider.value;
    }
    return sliderState;
  }

  clearSliderState() {
    for (let slider of this.faceSliders) {
      slider.value = 0;
    }
  }

  // If input alphas are not rounded, this will round them. Also filters the
  // input object to contain only keys for which there are sliders.
  roundTripSliderValues(alphaIdsToAlphas) {
    let toReturn = {};

    for (let alphaId in alphaIdsToAlphas) {
      let faceSlider = this.idsToSliders[alphaId];
      if (faceSlider === undefined) continue;
      else {
        faceSlider.value = alphaIdsToAlphas[alphaId];
        toReturn[parseFloat(alphaId)] = faceSlider.value;
      }
    }

    this._sliderState = this.calculateSliderState();

    return toReturn;
  }

  get areSlidersLocked() {
    return this._areSlidersLocked;
  }

  lockSliders() {
    this._areSlidersLocked = true;

    for (let slider of this.faceSliders) {
      slider.Disable();
    }
  }

  unlockSliders() {
    for (let slider of this.faceSliders) {
      slider.Enable();
    }

    this._areSlidersLocked = false;
  }

  // ------------------
  // Initial visibility
  // ------------------

  get areSlidersVisible() {
    return this.faceSliders[0].isAppearing;
  }
  showSliders() {
    let i = 0;
    for (let faceSlider of this.faceSliders) {
      window.setTimeout(() => { faceSlider.Appear(); }, i);
      i += 40;
    }
  }

  get isOutputVisible() {
    return this.imageOutput.isAppearing;
  }
  showOutputImage() {
    this.imageOutput.Appear();
  }

  // -------------------------------
  // Internal - Server Communication
  // -------------------------------

  // -----------------
  // Face manipulation
  // -----------------
  //
  // Manipulation types such as "Smiling" or "Beard" are indentified by integers
  // on the server and are hard-coded here in the client. These are the keys in
  // the manipulation argument object. Values for these keys should be from
  // 0 to 1.

  prepareImageManipulation() {
    this.imageSelector.StartLoadingVisual();
    this.imageOutput.StartLoadingVisual();
    this.imageSelector.Lock();
    this.lockSliders();

    let image = this.currentInputImage;
    this.currentInputVector = null;

    let imageEncodingCacheName = this.getImageEncodingCacheName(image);
    let cachedImageDataAndVector = this.imageDataAndVectorCache[
      imageEncodingCacheName
    ];
    if (cachedImageDataAndVector) {
      //console.log(imageEncodingCacheName + " already in cache. "
      // + "(prepareImageManipulation)");

      let imageData = cachedImageDataAndVector[0];
      let latentVector = cachedImageDataAndVector[1];

      // Note: Currently should be all zeroes.
      let sliderState = this.calculateSliderState();
      this.showAndCacheOutput(imageData, image, sliderState, latentVector);
      this.currentInputVector = latentVector;
    }
    else {
      //console.log(imageEncodingCacheName + " not present in cache "
      //  + "(prepareImageManipulation)");

      let post = ServerOps.postAPI;

      if (image instanceof UserImageChoice) {
        let userSlotIdx = image.userSlotIdx;

        let userSlotData = this.getUserSlotData(userSlotIdx);
        if (userSlotData) {
          //console.log("Data exists in user slot " + userSlotIdx);
        }
        else {
          //console.log("No data in user slot, asking for it");
          this.requestUserFileForEncoding(image);

          // We can't detect any change if the user doesn't change the input
          // file, so just unlock.
          this.imageOutput.StopLoadingVisual();
          this.imageSelector.StopLoadingVisual();
          this.imageSelector.Unlock();
        }
      }
      else {
        // Preset image, use DOM img.
        this.requestEncodingFromImg(image, image.domElement,
          (image, response) => { this.onImageEncodingResponse(image, response); });
      }
    }
  }

  requestUserFileForEncoding(userImage) {
    let image = userImage;
    image.RequestUserFile(
      (inputFile) => {
        var loadingImage = loadImage(
          inputFile,
          ((exifAdjustedImgAsCanvas) => {
            this.requestEncodingFromImg(
              image,
              exifAdjustedImgAsCanvas.toDataURL("image/png"),
              (image, response) => {
                this.onImageEncodingResponse(image, response, {
                  clearImageCacheOnSuccess: true
                });
              },
              { alreadyDataUrl: true }
            );
          }).bind(this),
          {
            orientation: true,
            maxWidth: MAX_IMAGE_SIZE,
            maxHeight: MAX_IMAGE_SIZE,
            contain: true,
            crossOrigin: "anonymous"
          }
        );
        if (loadingImage) {
          this.imageOutput.StartLoadingVisual();
          this.imageSelector.StartLoadingVisual();
          this.imageSelector.Lock();
        }
        else {
          console.error("Error loading image file.");
        }
      },
      (errResponse) => {
        console.error(errResponse);
      }
    );
  }

  onImageEncodingResponse(inputImage, response, options) {
    //console.log("response from server: ");
    //console.log(response);

    if (!response["face_found"]) {
      // No face in this image.
      this.imageSelector.Unlock();
      this.imageSelector.StopLoadingVisual();
      this.imageSelector.ShowNoFaceFoundPopup();
      this.imageOutput.StopLoadingVisual();
      return;
    }

    let image = inputImage;
    let imageData = response.img[0];
    let latentVector = response.z[0];
    let manipulationValuesFromServer = response.proj[0];
    let sliderState = this.calculateSliderState();

    if (options && options.clearImageCacheOnSuccess) {
      this.clearCacheForImage(image);
    }

    let alphaIdsToRoundedAlphas =
      this.roundTripSliderValues(manipulationValuesFromServer);
    this.clearSliderState();

    let imageManipulationString = this.getImageManipulationString(
      image, alphaIdsToRoundedAlphas
    );

    // Set the output image to the retrieved image data.
    this.showAndCacheOutput(imageData, image, sliderState, latentVector);

    this.imageSelector.StopLoadingVisual();
    this.imageSelector.Unlock();
    image.SetImageSource(imageData);

    // Set the current input vector now that we know have it from the
    // server. This is sent back to the server for manipulation
    // requests.
    this.currentInputVector = latentVector;

    let oneTimeImageLoadedCallback = this.oneTimeImageLoadedCallback;
    if (oneTimeImageLoadedCallback) {
      oneTimeImageLoadedCallback();

      this.oneTimeImageLoadedCallback = null;
    }
  }

  showAndCacheOutput(imageData, inputImage, sliderState, inputImageLatentVector) {
    // Set output image to display the image data.
    this.imageOutput.SetImageSource(imageData, inputImage.name);

    // Cache vector and image data for this slider state and input image.
    let imageManipString = this.getImageManipulationString(inputImage, sliderState);
    let cachedData = this.imageDataAndVectorCache[imageManipString];
    if (cachedData) {
      //console.log("Image and vector already in cache for " + imageManipString);
    }
    else {
     ("Adding manip string to cache: " + imageManipString);
      this.imageDataAndVectorCache[imageManipString] =
        [imageData, inputImageLatentVector];

      ensureListExistsAtKey(this.imageCacheEntries, inputImage.name);
      this.imageCacheEntries[inputImage.name].push(imageManipString);
    }

    // Handle visual changes common to receiving output data from the server.
    if (!this.areSlidersVisible) {
      window.setTimeout(() => { this.showSliders(); }, 400);
    }
    if (!this.isOutputVisible) {
      this.showOutputImage();
    }
    if (this.imageOutput.isLoadingVisual) {
      this.imageOutput.StopLoadingVisual();
    }
    if (this.areSlidersLocked) {
      this.unlockSliders();
    }
    this.imageSelector.Unlock();
    this.imageSelector.StopLoadingVisual();

    if (this.onNextOutputShown) {
      this.onNextOutputShown();

      this.onNextOutputShown = null;
    }
  }

  getImageManipulationString(image, alphaIdsToRoundedAlphas) {
    let toReturn = image.name;

    // The faceManipulations IDs correspond to indexes returned from the server
    // along those manipulation "vectors".
    for (let manipulationName in this.faceManipulations) {
      if (alphaIdsToRoundedAlphas) {
        toReturn += "_" + alphaIdsToRoundedAlphas[
          this.faceManipulations[manipulationName] // name to alpha ID
        ];
      }
      else {
        toReturn += "_" + 0;
      }
    }

    //console.log("Returning manipulation string: " + toReturn);

    return toReturn;
  }

  idsAndAlphasFromSliderState(sliderState) {
    let alphaIds = [];
    let alphas = [];
    for (let alphaId in sliderState) {
      alphaIds.push(parseFloat(alphaId));
      alphas.push(parseFloat(sliderState[alphaId]));
    }
    return [alphaIds, alphas];
  }

  requestManipulation(sliderState) {
    let image = this.imageSelector.currentSelectedImage;
    if (image === null) {
      console.error("Manipulation request, but no image was selected.");
    }
    let imageEncodingCacheName = this.getImageEncodingCacheName(
      image
    );
    let cachedImageDataAndVector = this.imageDataAndVectorCache[imageEncodingCacheName];
    if (!cachedImageDataAndVector) {
      console.error("No cached encoding available for " + imageEncodingCacheName);
      return;
    }
    let inputVector = cachedImageDataAndVector[1];

    sliderState = this.roundTripSliderValues(sliderState);
    let [alphaIds, alphas] = this.idsAndAlphasFromSliderState(sliderState);
    let manipString = this.getImageManipulationString(image, sliderState);

    let imageVectorPair = this.imageDataAndVectorCache[manipString];
    if (imageVectorPair === undefined) {
      //console.log("No cache present for manip string: " + manipString);

      let isPreset = !(image instanceof UserImageChoice);
      if (isPreset) {
        // Known preset image, we can request this manipulation from s3.
        let cachedImgUrl = ServerOps.getCachedManipUrl(image.presetName,
          alphaIds, alphas);

        toDataURL(cachedImgUrl, (imageData) => {
          this.showAndCacheOutput(imageData, image, sliderState, inputVector);
        });
      }
      else {
        let post = ServerOps.postAPI;
        post(
          "/api/manipulate_all",
          {
            z: inputVector,
            typs: alphaIds,
            alphas: alphas
          },
          (response) => {
            let imageData = response.img[0];

            this.showAndCacheOutput(imageData, image, sliderState, inputVector);
          },
          (errResponse) => {
            console.error(errResponse);
          }
        );
      }
    }
    else {
      //console.log("already in cache! " + manipString);
      this.showAndCacheOutput(imageVectorPair[0], image, sliderState,
        inputVector);
    }
  }

}

// ==============
// Image Selector
// ==============

const IMAGE_SIZE = 59;
const SELECTED_SIZE = 178;
const FRAME_SIZE = 178;
const IMAGE_MARGIN = 1;

const SPEED = 6;

class ImageSelector {

  constructor(container, onImageSelected, imageRequestsUserUploadCallback, options) {
    this._isOpen = true;
    this.container = container;
    this.onImageSelected = onImageSelected;
    this._size = FRAME_SIZE;

    let label = $(document.createElement("div"));
    let labelName = (options && options.labelName) || "Input";
    label.html(labelName);
    let labelClass = (options && options.labelClass) || "GlowDemo_InputLabel";
    label.addClass("GlowDemo");
    label.addClass(labelClass);
    this.container.append(label);
    this.label = label;

    let element = $(document.createElement("div"));
    element.addClass("GlowDemo");
    element.addClass("GlowDemo_ImageFrame");
    this.container.append(element);
    this.element = element;

    // Image selector images
    let cersei    = new ImageChoice("Lena", "cersei",
      `${IMAGE_SRC_PREFIX}/cersei.png`); // + '?' + (new Date()).getTime());
    let leo   = new ImageChoice("Leo", "leo",
      `${IMAGE_SRC_PREFIX}/leo.png`); // + '?' + (new Date()).getTime());
    let rashida = new ImageChoice("Rashida", "rashida",
      `${IMAGE_SRC_PREFIX}/rashida.png`); // + '?' + (new Date()).getTime());
    let neil    = new ImageChoice("Neil", "neil",
      `${IMAGE_SRC_PREFIX}/neil.png`); // + '?' + (new Date()).getTime());
    let beyonce = new ImageChoice("Beyoncé", "beyonce",
      `${IMAGE_SRC_PREFIX}/beyonce.png`); // + '?' + (new Date()).getTime());
    let geoff   = new ImageChoice("Geoff", "geoff",
      `${IMAGE_SRC_PREFIX}/geoff.png`); // + '?' + (new Date()).getTime());

    this.imageRequestsUserUploadCallback = imageRequestsUserUploadCallback;
    let userSlot0 = new UserImageChoice("User Slot 0", element, {
      editClickCallback: () => {
        // To re-upload an image, just clear its encoding cache and
        // act like it was selected.
        let userSlotImage = this.images[6];
        this.imageRequestsUserUploadCallback(userSlotImage);
        //this.onImageSelected(userSlotImage);
      }
    });
    let userSlot1 = new UserImageChoice("User Slot 1", element, {
      editClickCallback: () => {
        // To re-upload an image, just clear its encoding cache and
        // act like it was selected.
        let userSlotImage = this.images[7];
        this.imageRequestsUserUploadCallback(userSlotImage);
        //this.onImageSelected(userSlotImage);
      }
    });
    let userSlot2 = new UserImageChoice("User Slot 2", element, {
      editClickCallback: () => {
        // To re-upload an image, just clear its encoding cache and
        // act like it was selected.
        let userSlotImage = this.images[8];
        this.imageRequestsUserUploadCallback(userSlotImage);
        //this.onImageSelected(userSlotImage);
      }
    });

    this.images = [
      cersei,     leo,      rashida,
      neil,       beyonce,    geoff,
      userSlot0,  userSlot1,  userSlot2
    ];
    for (let image of this.images) {
      this.element.append(image.element);
      window.setTimeout(() => { image.selector = this; });
    }
    this.selectedIdx = 4;

    // First-open callback
    this.firstOpenCallback = null;
    this._seenFirstOpen = false;

    // Opacity (lock/unlock state).
    this._opacity = 1;

    // Set up a loading visual element to hover over the image frame.
    let loadingVisual = new LoadingVisual(this.element);
    this.loadingVisual = loadingVisual;

    // Timer to drive animation of a "no face found" overlay.
    // Off by default.
    this._noFaceFoundPopupTimer = 100000;
    let noFaceFoundPopUpElement = $(document.createElement("div"));
    noFaceFoundPopUpElement.html(
      "No face was found in the uploaded image.<br><br>Try to get a face as well-lit "
      + "as possible."
    );
    noFaceFoundPopUpElement.addClass("GlowDemo_ImageSelectorNoFaceFoundOverlay");
    noFaceFoundPopUpElement.addClass("GlowDemo");
    this.container.append(noFaceFoundPopUpElement);
    this.noFaceFoundPopUpElement = noFaceFoundPopUpElement;
    this._noFaceFoundPopupOpacity = 0;
    this.noFaceFoundPopUpElement.css("opacity", 0);
  }

  Lock() {
    this._locked = true;
  }

  Unlock() {
    this._locked = false;
  }

  StartLoadingVisual() {
    this.loadingVisual.Appear();
  }
  StopLoadingVisual() {
    this.loadingVisual.Hide();
  }

  ShowNoFaceFoundPopup() {
    this._noFaceFoundPopupTimer = 0;
  }

  Update() {
    let targetSize = IMAGE_SIZE;
    if (this.isOpen) {
      let centerOffset = FRAME_SIZE / 2 - IMAGE_SIZE / 2;
      for (let i = 0; i < 9; i++) {
        this.images[i].targetPosition = [
          ((i % 3) - 1) * (IMAGE_SIZE + IMAGE_MARGIN) + centerOffset,
          (Math.floor(i / 3) - 1) * (IMAGE_SIZE + IMAGE_MARGIN) + centerOffset
        ];
      }
      targetSize = FRAME_SIZE;

      this.images[this.selectedIdx].targetSize = IMAGE_SIZE;
    }
    else {
      for (let image of this.images) {
        image.targetPosition = [
          SELECTED_SIZE / 2 - IMAGE_SIZE / 2,
          SELECTED_SIZE / 2 - IMAGE_SIZE / 2
        ];
        image.targetSize = IMAGE_SIZE;
      }
      targetSize = FRAME_SIZE;

      // Center the selected image in the frame.
      let centerOffset = FRAME_SIZE / 2 - SELECTED_SIZE / 2;
      this.images[this.selectedIdx].targetPosition = [centerOffset, centerOffset];
      this.images[this.selectedIdx].targetSize = SELECTED_SIZE;
    }

    // Size update.
    this._size = lerp(this._size, targetSize, SPEED * DELTA_TIME);
    if (this._size != targetSize) {
      this.element.css("width", this._size);
      this.element.css("height", this._size);
    }

    // Update image choices and their z indices.
    for (let image of this.images) {
      image.Update();
      image.element.css("z-index", 0);
    }
    this.images[this.selectedIdx].element.css("z-index", 1);

    // Opacity update (lock/unlock state visibility).
    let targetOpacity = 1;
    if (this._locked) targetOpacity = 0.4;
    if (Math.abs(this._opacity - targetOpacity) > 0.00001) {
      this._opacity = lerp(this._opacity, targetOpacity, 4 * DELTA_TIME);
      this.element.css("opacity", this._opacity);
    }

    // Update the loading visual.
    this.loadingVisual.Update();

    // Update the no face found visual.
    let targetNoFaceFoundPopUpOpacity = 0;
    let noFaceFoundPopupDuration = 20;
    if (this._noFaceFoundPopupTimer < noFaceFoundPopupDuration) {
      this._noFaceFoundPopupTimer += DELTA_TIME;

      targetNoFaceFoundPopUpOpacity = 1;
    }
    if (Math.abs(
        this._noFaceFoundPopupOpacity - targetNoFaceFoundPopUpOpacity) > 0.00001) {
      this._noFaceFoundPopupOpacity
       = lerp(this._noFaceFoundPopupOpacity, targetNoFaceFoundPopUpOpacity,
              4 * DELTA_TIME);
      this.noFaceFoundPopUpElement.css("opacity", this._noFaceFoundPopupOpacity);
    }
    if (this._noFaceFoundPopupOpacity < 0.01) {
      this.noFaceFoundPopUpElement.css("display", "none");
    }
    if (this._noFaceFoundPopupOpacity > 0.01) {
      this.noFaceFoundPopUpElement.css("display", "initial");
    }
  }

  get isOpen() { return this._isOpen; }
  get isClosed() { return !this._isOpen; }
  Open() { this._isOpen = true; }
  Close() { this._isOpen = false; }

  get currentSelectedImage() {
    if (this.isOpen) return null;
    else {
      return this.images[this.selectedIdx];
    }
  }

  isImageSelected(image) {
    return this.currentSelectedImage === image;
  }

  onImageClicked(image) {
    if (this._locked) return;

    let clickedIndex = -1;
    for (let i = 0; i < 9; i++) {
      if (this.images[i] === image) {
        clickedIndex = i; break;
      }
    }
    if (clickedIndex == -1) {
      console.error("Image reported click to a selector it's not a part of.");
    }
    else {
      if (this.isOpen) {
        this.selectedIdx = clickedIndex;
        this.Close();

        this.onImageSelected(image);
      }
      else {
        if (!this._seenFirstOpen && this.firstOpenCallback != null) {
          this.firstOpenCallback();

          this._seenFirstOpen = true;
        }

        this.Open();
      }
    }
  }

}

// ===========
// ImageChoice
// ===========
//
// An interactive wrapper around DOM images in the image selector.

class ImageChoice {

  constructor(name, presetName, imageSource) {
    this.name = name;
    this.presetName = presetName;

    let img = document.createElement("img");
    img.src = imageSource;
    img.alt = name;
    this.element = $(img);
    //this.element.attr("crossorigin", "anonymous");
    this.domElement = img;

    this.element.addClass("GlowDemo");
    this.element.addClass("GlowDemo_ImageChoice");

    this.targetPosition = [0, 0];
    this._position = [0, 0];

    this.targetSize = IMAGE_SIZE;
    this._size = IMAGE_SIZE;

    this.selector = null;
    this.element.click(() => { this.onClick(); });

    this.element.css("width", this._size);
    this.element.css("height", this._size);
    this.element.css("position", "absolute");

    // Name image data sharing.
    if (window.GlowDemo_Cache === undefined) {
      window.GlowDemo_Cache = {};
    }
    let imageChoicesForName = this.imageChoicesForName;
    if (imageChoicesForName === undefined) {
      this.imageChoicesForName = imageChoicesForName = [];
    }
    imageChoicesForName.push(this);
  }

  get imageChoicesForName() {
    return window.GlowDemo_Cache[`ImageChoice_${this.name}`];
  }
  set imageChoicesForName(value) {
    window.GlowDemo_Cache[`ImageChoice_${this.name}`] = value;
  }

  get position() {
    return this._position;
  }
  set position(value) {
    if (this._position[0] != value[0]) {
      this._position[0] = value[0];
      this.element.css("margin-left", value[0]);
    }
    if (this._position[1] != value[1]) {
      this._position[1] = value[1];
      this.element.css("margin-top", value[1]);
    }
  }

  Update() {
    if (Math.abs(this._position[0] - this.targetPosition[0]) < 1.01
        && Math.abs(this._position[1] - this.targetPosition[1]) < 1.01) {
      this.position = this.targetPosition;
    }
    else {
      this.position = lerp2(this._position, this.targetPosition,
        SPEED * DELTA_TIME);
    }

    this._size = lerp(this._size, this.targetSize, SPEED * DELTA_TIME);
    if (this._size != this.targetSize) {
      this.element.css("width", this._size);
      this.element.css("height", this._size);
    }
  }

  SetImageSource(src, options) {
    if (!(options) || (options.propagate === undefined)) {
      let imageChoicesForName = this.imageChoicesForName;
      if (imageChoicesForName !== undefined) {
        let propagateCount = 0;
        for (let imageChoice of this.imageChoicesForName) {
          if (imageChoice !== this) {
            imageChoice.SetImageSource(src, { propagate: false });
            propagateCount += 1;
          }
        }
        //console.log("propagateCount " + propagateCount);
      }
      else {
        console.error("No image choices for our name");
      }
    }

    this.domElement.src = src;
  }

  onClick() {
    let selector = this.selector;
    if (selector != null) {
      selector.onImageClicked(this);
    }
  }

}

// ===============
// UserImageChoice
// ===============
//
// As ImageChoice, but gives the user the ability to upload their own source image.

class UserImageChoice extends ImageChoice {

  constructor(name, container, options) {
    super(name, null, `${IMAGE_PLACEHOLDER_SRC}`); //+ '?' +(new Date()).getTime());

    this.container = container;

    // "User Slot X"
    this.userSlotIdx = parseInt(name);

    let fileInputElement = $(document.createElement("input"));
    fileInputElement.attr("type", "file");
    let isFacebookApp = function () {
      let ua = navigator.userAgent || navigator.vendor || window.opera;
      return (ua.indexOf("FBAN") > -1) || (ua.indexOf("FBAV") > -1);
    }();

    if (!isFacebookApp) {
      fileInputElement.attr("accept", "image/*");
    }

    fileInputElement.on("change", (ev) => { this.onFileChanged(ev);  });
    this.fileInputElement = fileInputElement;

    // Add an element on selected UserImageChoices to edit the choice even after
    // something has been uploaded to the slot.
    let EDIT_IMAGE_SRC = `${IMAGE_SRC_PREFIX}/EditIcon.png`;
    let editButton = new FadeImage(
      this.container,
      EDIT_IMAGE_SRC,
      {
        extraClasses: ["GlowDemo_UserImageEditButton"]
      }
    );
    this.editButton = editButton;
    this.container.mousemove(() => {
      if (this.isSelected) {
        this.editButton.Show();
        this.editButton.SetHideTimer(8);
      }
    });
    this.container.hover(() => {
      if (this.isSelected) {
        this.editButton.Show();
        this.editButton.SetHideTimer(8);
      }
    });

    if (options && options.editClickCallback) {
      this.editButton.element.click(() => {
        options.editClickCallback();
      });
    }
    else {
      this.editButton.element.click(() => {
        console.log("Edit button clicked, but no callback was provided.");
      });
    }
  }

  get isSelected() {
    return this.selector.isImageSelected(this);
  }

  onClick() {
    super.onClick();

    if (this.selector != null) {
      if (this.isSelected) {
        this.editButton.Show();
        this.editButton.SetHideTimer(8);
      }
    }
  }

  Update() {
    super.Update();

    if (!this.isSelected) {
      this.editButton.Hide();
    }
    this.editButton.Update();
  }

  RequestUserFile(fileChangedCallback) {
    this.fileChangedCallback = fileChangedCallback;

    this.fileInputElement.trigger("click");
  }

  onFileChanged(ev) {
    //console.log("File changed:");
    //console.log(ev);

    this.hasFile = true;

    let input = ev.target;
    if (input.files && input.files[0]) {
      //console.log("got an input file, " + input.files[0].size + " bytes in size");
      this.fileChangedCallback(input.files[0]);
    }
    else {
      console.error("No input file");
    }
  }

}

// ============
// Slider Frame
// ============
//
// Manages slider loading, visibility, and control.

// class SliderFrame {

//   constructor(container, faceManipulations) {

//   }

// }

// ===========
// Face Slider
// ===========

class FaceSlider {

  constructor(container, name, faceManipulationId, startVisible, options) {
    this.name = name;
    this.faceManipulationId = faceManipulationId;

    // Slider parent element.
    let divElement = $(document.createElement("div"));
    divElement.addClass("GlowDemo");
    divElement.addClass("GlowDemo_FaceSliderContainer");
    container.append(divElement);
    this.divElement = divElement;

    // Slider label element.
    let label = $(document.createElement("div"));
    label.html("<p>" + name + "</p>");
    label.addClass("GlowDemo");
    label.addClass("GlowDemo_FaceSliderLabel");
    if (options && options.extraLabelClasses) {
      for (let extraLabelClass of options.extraLabelClasses) {
        label.addClass(extraLabelClass);
      }
    }
    let placeLabelBefore = !options || !(options.placeLabelAfter);
    if (placeLabelBefore) {
      divElement.append(label);
    }
    this.label = label;

    // Slider <input> element.
    let inputDivElement = $(document.createElement("div"));
    inputDivElement.addClass("GlowDemo");
    inputDivElement.addClass("GlowDemo_FaceSlider");
    let element = $(document.createElement("input"));
    if (options && options.extraSliderClasses) {
      for (let extraSliderClass of options.extraSliderClasses) {
        element.addClass(extraSliderClass);
      }
    }
    inputDivElement.append(element);
    divElement.append(inputDivElement);
    let sliderAttributes = {};
    if (!options || !options.sliderAttributes) {
      sliderAttributes = {
        "type": "range",
        "min": "-0.99",
        "max": "0.99",
        "value": "0",
        "step": "0.33"
      };
    }
    else {
      sliderAttributes = options.sliderAttributes;
    }
    element.attr(sliderAttributes);
    this.element = element;
    this.domElement = element.get(0);

    if (!placeLabelBefore) {
      divElement.append(label);
    }

    // Visibility control.
    let sliderHider = $(document.createElement("div"));
    sliderHider.addClass("GlowDemo");
    sliderHider.addClass("GlowDemo_SliderHider");
    if (options && options.extraHiderClasses) {
      for (let extraHiderClass of options.extraHiderClasses) {
        sliderHider.addClass(extraHiderClass);
      }
    }
    divElement.append(sliderHider);
    this.sliderHider = sliderHider;

    if (startVisible) {
      this._visibleAmount = 0.99;
      this._targetVisibleAmount = 1;
    }
    else {
      this._visibleAmount = 0.01;
      this._targetVisibleAmount = 0;
    }

    // Interaction state.
    this._isHeld = false;

    this.element.mousedown((ev) => { this.onMouseDown(ev); });
    this.element.mouseup((ev) => { this.onMouseUp(ev); });
    this.element.on('touchstart', (ev) => { this.onMouseDown(ev); });
    this.element.on('touchend', (ev) => { this.onMouseUp(ev); });

    // this.element.on("input", (ev) => { this.onMouseDown(ev); });
    // this.element.on("change", (ev) => { this.onMouseUp(ev); });


    // Enable/Disable state.
    this._isEnabled = true;
    this._opacity = 1;
  }

  get value() { return this.domElement.value; }
  set value(value) { this.domElement.value = value; }

  get isAppearing() { return this._targetVisibleAmount > 0; }
  Appear() { this._targetVisibleAmount = 1; }
  get isHiding() { return this._targetVisibleAmount < 1; }
  Hide() { this._targetVisibleAmount = 0; }

  get isHeld() {
    return this._isEnabled  && (this._isHeld || this._heldTimer < 4);
  }

  get isEnabled() { return this._isEnabled; }
  Enable() {
    this._isEnabled = true;

    this.domElement.removeAttribute("disabled");
  }
  Disable() {
    this._isEnabled = false;

    this.domElement.setAttribute("disabled", "disabled");
  }

  ForceHide() {
    this._forceHide = true;
  }
  UnForceHide() {
    this._forceHide = false;
  }

  Update() {
    // Visibility update.
    if (this._visibleAmount != this._targetVisibleAmount) {
      this._visibleAmount =
        lerp(this._visibleAmount, this._targetVisibleAmount, 0.4 * DELTA_TIME);
      this.divElement.css("opacity", `${this._visibleAmount}`);
    }
    if (this._visibleAmount > 0.8) {
      this.sliderHider.css("display", "none");
    }
    if (this._visibleAmount < 0.01) {
      this.sliderHider.css("display", "block");
    }

    // Enable/disable update.
    let targetOpacity = 1;
    if (!this._isEnabled) { targetOpacity = 0.4; }
    if (targetOpacity != this._opacity) {
      this._opacity = lerp(this._opacity, targetOpacity, 3 * DELTA_TIME);
      this.element.css("opacity", this._opacity);
      this.label.css("opacity", this._opacity);
    }

    if (this._heldTimer < 200) {
      this._heldTimer += 1;
    }

    if (this._forceHide) {
      this.element.css("opacity", 0);
      this._visibleAmount = 0;
    }
  }

  onMouseDown(ev) {
    this._isHeld = true;
    this._heldTimer = 0;
  }

  onMouseUp(ev) {
    this._isHeld = false;
  }

}

// ============
// Image Output
// ============

class ImageOutput {

  constructor(container, options) {
    this.container = container;

    // Load output image label.
    let label = $(document.createElement("div"));
    let labelName = (options && options.labelName) || "Output";
    label.html(labelName);
    let labelClass = (options && options.labelClass) || "GlowDemo_OutputLabel";
    label.addClass(labelClass);
    this.container.append(label);
    this.label = label;

    // Load output image frame.
    let imageFrame = $(document.createElement("div"));
    imageFrame.addClass("GlowDemo");
    imageFrame.addClass("GlowDemo_ImageFrame");
    imageFrame.addClass("GlowDemo_OutputImageFrame");
    imageFrame.css("width", "178px");
    imageFrame.css("height", "178px");
    this.container.append(imageFrame);
    this.imageFrame = imageFrame;

    // Load output image.
    let imageElement = $(document.createElement("img"));
    imageElement.addClass("GlowDemo");
    imageElement.addClass("GlowDemo_OutputImage");
    imageElement.attr("src", IMAGE_PLACEHOLDER_SRC);
    let imageElementDiv = $(document.createElement("div"));
    imageFrame.append(imageElementDiv);
    imageElementDiv.append(imageElement);
    this.imageElement = imageElement;

    // Load an overlay element to hide the output image on start.
    let outputHider = $(document.createElement("div"));
    outputHider.addClass("GlowDemo");
    outputHider.addClass("GlowDemo_OutputHider");
    this.container.append(outputHider);
    this.outputHider = outputHider;

    // Load a download button element.
    let downloadButton = new FadeImage(
      imageElementDiv,
      DOWNLOAD_IMAGE_SRC,
      {
        extraClasses: ["GlowDemo_DownloadButton"]
      }
    );
    this.downloadButton = downloadButton;
    this.container.mousemove(() => {
      this.downloadButton.Show();
      this.downloadButton.SetHideTimer(8);
    })
    this.container.hover(() => {
      this.downloadButton.Show();
      this.downloadButton.SetHideTimer(8);

      if (this.firstClickOrHover) {
        this.firstClickOrHover();
        this.firstClickOrHover = undefined;
      }
    })
    this.container.click(() => {
      this.downloadButton.Show();
      this.downloadButton.SetHideTimer(8);

      if (this.firstClickOrHover) {
        this.firstClickOrHover();
        this.firstClickOrHover = undefined;
      }
    });
    if (options && options.downloadClickCallback) {
      this.downloadButton.element.click(() => {
        if (options && options.preDownloadClickCallback) {
          options.preDownloadClickCallback();
        }

        this.downloadButton.ForceHide();

        window.setTimeout(() => {

          window.setTimeout(() => {
            this.downloadButton.UnForceHide();
            this.downloadButton.Show();
            this.downloadButton.SetHideTimer(8);
          }, 500);

          options.downloadClickCallback();
        }, 250);
      });
    }
    else {
      this.downloadButton.element.click(() => {
        console.log("Download button clicked, but no container to capture "
          + "was provided.");
      });
    }

    // Visibility control.
    if ((!options) || options.startVisible) {
      this._visibleAmount = 0.99;
      this._targetVisibleAmount = 1;
    }
    else {
      this._visibleAmount = 0.01;
      this._targetVisibleAmount = 0;
    }

    // Set up a loading visual element to hover over the image frame.
    let loadingVisual = new LoadingVisual(this.imageFrame);
    this.loadingVisual = loadingVisual;

    this.firstDisplayCallback = null;
    this._seenFirstDisplay;
  }

  get isAppearing() { return this._targetVisibleAmount > 0; }
  Appear() { this._targetVisibleAmount = 1; }
  get isHiding() { return this._targetVisibleAmount < 1; }
  Hide() { this._targetVisibleAmount = 0; }

  Update() {
    // Visibility update.
    if (this._visibleAmount != this._targetVisibleAmount) {
      this._visibleAmount =
        lerp(this._visibleAmount, this._targetVisibleAmount, 2 * DELTA_TIME);
      this.outputHider.css(
        "background-color",
        `rgba(255, 255, 255, ${(1 - this._visibleAmount)}`
      );
    }

    if (this._visibleAmount > 0.95) {
      this.outputHider.css("display", "none");
    }
    if (this._visibleAmount < 0.01) {
      this.outputHider.css("display", "block");
    }

    // Update the loading visual.
    this.loadingVisual.Update();

    if (this.loadingVisual.isAppearing && this.downloadButton.isShowing) {
      this.downloadButton.Hide();
    }

    // Update the download button.
    this.downloadButton.Update();
  }

  SetImageSource(src, name) {
    this.imageName = name;
    this.imageElement.get(0).src = src;

    if (this.firstDisplayCallback && !this._seenFirstDisplay) {
      this.firstDisplayCallback();

      this._seenFirstDisplay = true;
    }
  }

  StartLoadingVisual() {
    this.loadingVisual.Appear();
  }
  get isLoadingVisual() {
    return this.loadingVisual.isAppearing;
  }
  StopLoadingVisual() {
    this.loadingVisual.Hide();
  }

}

// ===============
// Utility Classes
// ===============

class LoadingVisual {

  constructor(container) {
    let element = $(document.createElement("img"));
    element.addClass("GlowDemo");
    element.addClass("GlowDemo_LoadingVisual");
    element.attr("src", LOADING_IMAGE_SRC); // + '?' +(new Date()).getTime());
    container.append(element);
    this.element = element;

    // Animating rotation.
    this._rotationAmount = 0;

    // Visibility control.
    this._targetVisibleAmount = 0;
    this._visibleAmount = 0;
    element.css("display", "none");

    element.css("width", 60);
    element.css("height", 60);
  }

  Update() {
    // Visibility update.
    if (this._visibleAmount != this._targetVisibleAmount) {
      // Make the transition from not-visible to slightly-visible just a tad
      // more delayed, to prevent the visual from flashing briefly when it
      // last long anyway.
      let limitSpeed = false;
      if (this._targetVisibleAmount > this._visibleAmount
          && this._visibleAmount < 0.01) {
        limitSpeed = true;
      }

      if (!limitSpeed) {
        this._visibleAmount =
          lerp(this._visibleAmount, this._targetVisibleAmount,
            SPEED * DELTA_TIME);
      }
      else {
        this._visibleAmount += 0.0001;
      }
      this.element.css("opacity", this._visibleAmount);
    }

    if (this._visibleAmount < 0.01) {
      this.element.css("display", "none");
    }
    if (this._visibleAmount > 0.01) {
      this.element.css("display", "block");
    }

    if (this._visibleAmount > 0.01) {
      this._rotationAmount += 30 * DELTA_TIME;
      this._rotationAmount %= 360;
      this.element.css("transform", `rotate(${this._rotationAmount}deg)`);
    }
  }

  Appear() {
    this._targetVisibleAmount = 1;
  }
  Hide() {
    this._targetVisibleAmount = 0;
  }
  get isAppearing() {
    return this._targetVisibleAmount > 0;
  }
  get isHiding() {
    return this._targetVisibleAmount < 1;
  }

}

class FadeImage {

  constructor(container, src, options) {
    let element = $(document.createElement("img"));
    element.addClass("GlowDemo");
    element.addClass("GlowDemo_FadeButton");
    if (options && options.extraClasses) {
      for (let extraClass of options.extraClasses) {
        element.addClass(extraClass);
      }
    }
    element.attr("src", src); // + '?' +(new Date()).getTime());
    container.append(element);
    this.element = element;

    // Visibility control.
    this._targetVisibleAmount = 0;
    this._visibleAmount = 0;
    element.css("display", "none");
  }

  Update() {
    if (this._disabled) {
      return;
    }

    // Visibility update.
    if (this._visibleAmount != this._targetVisibleAmount) {
       this._visibleAmount =
        lerp(this._visibleAmount, this._targetVisibleAmount,
          SPEED * DELTA_TIME);
      this.element.css("opacity", this._visibleAmount);
    }

    if (this._visibleAmount < 0.01) {
      this.element.css("display", "none");
    }
    if (this._visibleAmount > 0.01) {
      this.element.css("display", "block");
    }

    if (this._hideTimer !== undefined && this._hideTimer !== null) {
      this._hideTimer -= DELTA_TIME;
      if (this._hideTimer <= 0) {
        this._hideTimer = null;
        this.Hide();
      }
    }
  }

  Show() {
    this._targetVisibleAmount = 1;
  }
  Hide() {
    this._targetVisibleAmount = 0;
  }

  UnForceHide() {
    this._disabled = false;

    if (this._visibleAmount > 0.01) {
      this.element.css("display", "block");
    }
  }
  ForceHide() {
    this._disabled = true;
    this.element.css("display", "none");
  }

  get isShowing() {
    return this._targetVisibleAmount > 0;
  }
  get isHiding() {
    return this._targetVisibleAmount < 1;
  }

  SetHideTimer(time) {
    this._hideTimer = time;
  }

}

// =====
// Hints
// =====

class Hint {

  constructor(container, hintHtml, options) {
    let element = $(document.createElement("div"));
    element.addClass("GlowDemo");
    element.addClass("GlowDemo_Hint");
    if (options && options.extraClasses) {
      for (let extraClass of options.extraClasses) {
        element.addClass(extraClass);
      }
    }
    container.append(element);
    element.html(hintHtml);
    this.element = element;

    this._visibleAmount = 0.01;
    this.element.css("opacity", `0`);
    this._targetVisibleAmount = 1;

    // Wait a bit before appearing.
    this._waitTimer = 0;
    this._waitDuration = 600;

    if (options && options.startHidden) {
      this._targetVisibleAmount = 0;
      this._visibleAmount = 0;
      this.element.css("opacity", `0`);
    }
  }

  Hide() {
    this._targetVisibleAmount = 0;
  }

  Show() {
    this._targetVisibleAmount = 1;
  }

  SetHideTimer(time) {
    this._hideTimer = time;
  }

  Update() {
    if (this._waitTimer < this._waitDuration) {
      this._waitTimer += DELTA_TIME * 1000;
      return;
    }

    let wasVisible = this._visibleAmount >= 0.01;

    // Visibility update.
    if (this._visibleAmount != this._targetVisibleAmount) {
      this._visibleAmount =
        lerp(this._visibleAmount, this._targetVisibleAmount,
          SPEED * 0.4 * DELTA_TIME);
      this.element.css("opacity", `${this._visibleAmount}`);
    }

    if (this._visibleAmount < 0.01) {
      this.element.css("display", "none");

      if (wasVisible && this.onHiddenOnce) {
        this.onHiddenOnce();

        this.onHiddenOnce = null;
      }
    }
    if (this._visibleAmount >= 0.01) {
      this.element.css("display", "block");
    }

    if (this._hideTimer) {
      this._hideTimer -= DELTA_TIME;

      if (this._hideTimer <= 0) {
        this.Hide();

        this._hideTimer = null;
      }
    }
  }

}

// =================
// Utility Functions
// =================

function lerp(a, b, t) {
  t = Math.min(1, Math.max(0, t));
  let result = a * (1 - t) + (b * t);
  // This "snapping" behavior prevents this method from being a true lerp,
  // but results in fewer wasteful "almost there" calculations with the way
  // it is used to animate properties.
  if (Math.abs(result, b) < 0.00001) {
    return b;
  }
  return result;
}

function lerp2(a, b, t) {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];
}

function areContentsEqual(obj0, obj1) {
  if (obj0 === undefined || obj1 === undefined) return false;
  if (obj0 === null || obj1 === null) return false;

  let tested = {};
  for (let propName in obj0) {
    if (obj1[propName] != obj0[propName]) return false;
    tested[propName] = true;
  }
  for (let propName in obj1) {
    if (!(propName in tested)) return false;
  }

  return true;
}

function toDataURL(url, callback) {
  var xhr = new XMLHttpRequest();
  xhr.open('get', url + '?x-request=html'); // + '?' +(new Date()).getTime());
  xhr.responseType = 'blob';
  xhr.onload = function () {
    var fr = new FileReader();

    fr.onload = function (ev) {
      callback(ev.target.result);
    };

    fr.readAsDataURL(xhr.response); // async call
  };

  xhr.send();
}

function ensureListExistsAtKey(dict, key) {
  if (dict[key]) {
    return;
  }
  else {
    dict[key] = [];
  }
}
