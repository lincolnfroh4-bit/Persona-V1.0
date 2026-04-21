const state = {
  models: [],
  pipaPackages: [],
  qualityPresets: {},
  preprocessPipelines: [],
  masterOptions: null,
  generateOptions: null,
  files: [],
  indexFile: null,
  previewSourceId: null,
  previewSourceKey: "",
  previewDebounceHandle: null,
  previewRequestToken: 0,
  currentJobId: null,
  currentJobKind: "master-conversion",
  pollHandle: null,
  generateGuideFile: null,
  currentGenerateJobId: null,
  generatePollHandle: null,
  isolatorOptions: null,
  isolatorFiles: [],
  currentIsolatorJobId: null,
  isolatorPollHandle: null,
  masteringOptions: null,
  masteringSourceFile: null,
  masteringReferenceFiles: [],
  currentMasteringJobId: null,
  masteringPollHandle: null,
  optimizeOptions: null,
  optimizeFiles: [],
  currentOptimizeJobId: null,
  optimizePollHandle: null,
  albumOptions: null,
  albumProjects: [],
  currentAlbumProjectId: null,
  albumCurrentProject: null,
  albumBulkFiles: [],
  albumDragSongIndex: null,
  apiComposeOptions: null,
  apiComposeMidiFile: null,
  apiComposeBeatFile: null,
  currentApiComposeJobId: null,
  apiComposePollHandle: null,
  touchUpReferenceFile: null,
  touchUpSourceFile: null,
  currentTouchUpJobId: null,
  touchUpPollHandle: null,
  detagVoices: [],
  detagFile: null,
  currentDetagJobId: null,
  detagPollHandle: null,
  trainingFiles: [],
  trainingPlanFiles: [],
  trainingTranscriptFiles: [],
  trainingOptions: null,
  currentTrainingJobId: null,
  trainingPollHandle: null,
  trainingPollInFlight: false,
  currentConversionModelName: "",
  currentConversionModelLabel: "",
  currentConversionModelSystem: "",
  convertResultRetryCount: 0,
};

const PERSONA_PACKAGE_MODES = ["persona-v1", "persona-v1.1", "persona-lyric-repair", "persona-aligned-pth"];
const DOWNLOADABLE_TRAINING_PACKAGE_MODES = [
  "persona-v1",
  "persona-v1.1",
  "persona-lyric-repair",
  "persona-aligned-pth",
  "concert-remaster-paired",
  "classic-rvc-support",
];
const RESUMABLE_PERSONA_PACKAGE_MODES = ["persona-v1", "persona-v1.1", "persona-lyric-repair"];
const LEAD_BUILDER_PACKAGE_MODE = "persona-aligned-pth";

function isPersonaPackageMode(mode) {
  return PERSONA_PACKAGE_MODES.includes(String(mode || "").trim().toLowerCase());
}

function isDownloadableTrainingPackageMode(mode) {
  return DOWNLOADABLE_TRAINING_PACKAGE_MODES.includes(String(mode || "").trim().toLowerCase());
}

function isResumablePersonaPackageMode(mode) {
  return RESUMABLE_PERSONA_PACKAGE_MODES.includes(String(mode || "").trim().toLowerCase());
}

function getPersonaPackageLabel(mode) {
  const normalized = String(mode || "").trim().toLowerCase();
  if (normalized === "persona-v1.1") {
    return "Persona v1.1";
  }
  if (normalized === "persona-lyric-repair") {
    return "Persona lyric repair";
  }
  if (normalized === "concert-remaster-paired") {
    return "Concert remaster";
  }
  if (normalized === "classic-rvc-support") {
    return "Classic RVC + SUNO audition";
  }
  if (normalized === "persona-aligned-pth") {
    return "Paired aligned conversion";
  }
  return "Persona v1.0";
}

function isAlignedPthPackageMode(mode) {
  return String(mode || "").trim().toLowerCase() === LEAD_BUILDER_PACKAGE_MODE;
}

function isClassicRvcPackageMode(mode) {
  return String(mode || "").trim().toLowerCase() === "classic-rvc";
}

function isClassicSupportPackageMode(mode) {
  return String(mode || "").trim().toLowerCase() === "classic-rvc-support";
}

function isLeadBuilderConversionModel(model) {
  const kind = String(model?.kind || "").trim().toLowerCase();
  const packageMode = String(model?.package_mode || "").trim().toLowerCase();
  return kind === "model"
    || packageMode === LEAD_BUILDER_PACKAGE_MODE
    || isClassicSupportPackageMode(packageMode)
    || isClassicRvcPackageMode(packageMode);
}

function getLeadBuilderModels() {
  return (state.models || []).filter((model) => isLeadBuilderConversionModel(model));
}

function getPersonaRepairModels() {
  return (state.models || []).filter(
    (model) => String(model?.kind || "").trim().toLowerCase() !== "classic",
  );
}

function getLeadBuilderSelectedModel() {
  return (
    getLeadBuilderModels().find((model) => model.name === modelSelect?.value)
    || getLeadBuilderModels()[0]
    || null
  );
}

function stripAlignedPrefix(fileName, prefixes) {
  const safeName = String(fileName || "").trim();
  if (!safeName) {
    return "";
  }
  const lastDot = safeName.lastIndexOf(".");
  const stem = lastDot > 0 ? safeName.slice(0, lastDot) : safeName;
  const upperStem = stem.toUpperCase();
  const cleanedPrefixes = [...prefixes]
    .map((prefix) => String(prefix || "").trim().toUpperCase())
    .filter(Boolean)
    .sort((left, right) => right.length - left.length);
  let workingStem = stem;
  for (const prefix of cleanedPrefixes) {
    if (upperStem.startsWith(prefix)) {
      workingStem = stem.slice(prefix.length);
      break;
    }
  }
  return workingStem.replace(/^[ _.-]+/, "").trim().toLowerCase();
}

const TRAINING_MODE_CONFIG = {
  "classic-rvc-support": {
    label: "Classic RVC + SUNO audition",
    basePrefixes: ["BASE"],
    targetPrefixes: ["TARGET", "VOCALP"],
    sourcePrefixes: ["SUNO", "PREP", "PRE", "SUPPORT"],
    baseName: "BASE",
    targetName: "TARGET",
    sourceName: "SUNO",
    requiresPairs: false,
    summary: "Classic RVC + SUNO audition uses `BASE*` as the real target-voice truth, optional `TARGET*` extra truth clips, and `SUNO*` only for fixed checkpoint auditions.",
  },
  "persona-aligned-pth": {
    label: "Paired aligned conversion",
    basePrefixes: ["BASE"],
    targetPrefixes: ["TARGET", "VOCALP"],
    sourcePrefixes: ["SUNO", "PREP", "PRE"],
    baseName: "BASE",
    targetName: "TARGET",
    sourceName: "SUNO",
    requiresPairs: true,
    summary: "Paired aligned conversion uses `BASE*`, `TARGET*`, and `SUNO*` clips.",
  },
  "concert-remaster-paired": {
    label: "Concert remaster",
    basePrefixes: [],
    targetPrefixes: ["CD", "MASTER", "STUDIO"],
    sourcePrefixes: ["CONCERT", "LIVE"],
    baseName: "",
    targetName: "CD",
    sourceName: "CONCERT",
    requiresPairs: true,
    summary: "Concert remaster uses matched `CONCERT*` and `CD*` clips.",
  },
};

function getTrainingModeConfig(mode) {
  return TRAINING_MODE_CONFIG[String(mode || "").trim().toLowerCase()] || TRAINING_MODE_CONFIG["classic-rvc-support"];
}

function summarizeTrainingFiles(files, mode) {
  const config = getTrainingModeConfig(mode);
  const baseKeys = new Set();
  const targetKeys = new Set();
  const sourceKeys = new Set();
  let baseCount = 0;
  let targetCount = 0;
  let sourceCount = 0;

  for (const file of files || []) {
    const stem = String(file?.name || "").replace(/\.[^/.]+$/, "");
    const upperStem = stem.toUpperCase();
    if (config.basePrefixes.some((prefix) => upperStem.startsWith(prefix))) {
      baseCount += 1;
      const key = stripAlignedPrefix(file.name, config.basePrefixes);
      if (key) {
        baseKeys.add(key);
      }
      continue;
    }
    if (config.targetPrefixes.some((prefix) => upperStem.startsWith(prefix))) {
      targetCount += 1;
      const key = stripAlignedPrefix(file.name, config.targetPrefixes);
      if (key) {
        targetKeys.add(key);
      }
      continue;
    }
    if (config.sourcePrefixes.some((prefix) => upperStem.startsWith(prefix))) {
      sourceCount += 1;
      const key = stripAlignedPrefix(file.name, config.sourcePrefixes);
      if (key) {
        sourceKeys.add(key);
      }
    }
  }

  const matchedKeys = [...targetKeys].filter((key) => sourceKeys.has(key));
  const unmatchedTargets = [...targetKeys].filter((key) => !sourceKeys.has(key));
  const unmatchedSources = [...sourceKeys].filter((key) => !targetKeys.has(key));
  return {
    mode: String(mode || "").trim().toLowerCase() || "classic-rvc-support",
    label: config.label,
    baseCount,
    targetCount,
    sourceCount,
    matchedPairCount: matchedKeys.length,
    unmatchedTargets,
    unmatchedSources,
    summary: config.summary,
    baseName: config.baseName,
    targetName: config.targetName,
    sourceName: config.sourceName,
    requiresPairs: Boolean(config.requiresPairs),
  };
}

function summarizeAlignedTrainingFiles(files) {
  return summarizeTrainingFiles(files, "persona-aligned-pth");
}

function summarizeConcertRemasterTrainingFiles(files) {
  return summarizeTrainingFiles(files, "concert-remaster-paired");
}

const showConvertTabButton = document.getElementById("showConvertTab");
const showGenerateTabButton = document.getElementById("showGenerateTab");
const showIsolatorTabButton = document.getElementById("showIsolatorTab");
const showOptimizeTabButton = document.getElementById("showOptimizeTab");
const showAlbumsTabButton = document.getElementById("showAlbumsTab");
const showApiComposeTabButton = document.getElementById("showApiComposeTab");
const showMasteringTabButton = document.getElementById("showMasteringTab");
const showTouchUpTabButton = document.getElementById("showTouchUpTab");
const showDetagTabButton = document.getElementById("showDetagTab");
const showTrainingTabButton = document.getElementById("showTrainingTab");
const railConvertTabButton = document.getElementById("railConvertTab");
const railIsolatorTabButton = document.getElementById("railIsolatorTab");
const railMasteringTabButton = document.getElementById("railMasteringTab");
const railTrainingTabButton = document.getElementById("railTrainingTab");
const convertTab = document.getElementById("convertTab");
const generateTab = document.getElementById("generateTab");
const isolatorTab = document.getElementById("isolatorTab");
const masteringTab = document.getElementById("masteringTab");
const optimizeTab = document.getElementById("optimizeTab");
const albumsTab = document.getElementById("albumsTab");
const apiComposeTab = document.getElementById("apiComposeTab");
const touchUpTab = document.getElementById("touchUpTab");
const detagTab = document.getElementById("detagTab");
const trainingTab = document.getElementById("trainingTab");

const modelSelect = document.getElementById("modelSelect");
const pipaPackageSelect = document.getElementById("pipaPackageSelect");
const pipaPackageSummary = document.getElementById("pipaPackageSummary");
const generateModelSelect = document.getElementById("generateModelSelect");
const refreshModelsButton = document.getElementById("refreshModels");
const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const fileList = document.getElementById("fileList");
const outputMode = document.getElementById("outputMode");
const outputModeSummary = document.getElementById("outputModeSummary");
const secondaryModelField = document.getElementById("secondaryModelField");
const secondaryModelSelect = document.getElementById("secondaryModelSelect");
const blendPercentageField = document.getElementById("blendPercentageField");
const blendPercentage = document.getElementById("blendPercentage");
const blendPercentageValue = document.getElementById("blendPercentageValue");
const secondaryBlendPercentageValue = document.getElementById("secondaryBlendPercentageValue");
const indexFileInput = document.getElementById("indexFileInput");
const indexDropZone = document.getElementById("indexDropZone");
const indexFileList = document.getElementById("indexFileList");
const pitchPreset = document.getElementById("pitchPreset");
const customPitchField = document.getElementById("customPitchField");
const customPitch = document.getElementById("customPitch");
const qualityPreset = document.getElementById("qualityPreset");
const qualitySummary = document.getElementById("qualitySummary");
const outputFormat = document.getElementById("outputFormat");
const masterLyrics = document.getElementById("masterLyrics");
const masterProfile = document.getElementById("masterProfile");
const masterProfileSummary = document.getElementById("masterProfileSummary");
const preprocessMode = document.getElementById("preprocessMode");
const preprocessModeSummary = document.getElementById("preprocessModeSummary");
const preprocessStrength = document.getElementById("preprocessStrength");
const startButton = document.getElementById("startButton");
const statusCard = document.getElementById("statusCard");
const statusTitle = document.getElementById("statusTitle");
const statusMessage = document.getElementById("statusMessage");
const progressBar = document.getElementById("progressBar");
const previewStatusCard = document.getElementById("previewStatusCard");
const previewStatusTitle = document.getElementById("previewStatusTitle");
const previewStatusMessage = document.getElementById("previewStatusMessage");
const previewProgressBar = document.getElementById("previewProgressBar");
const previewMeta = document.getElementById("previewMeta");
const previewPlayer = document.getElementById("previewPlayer");
const resultSummary = document.getElementById("resultSummary");
const results = document.getElementById("results");
const modelEmptyState = document.getElementById("modelEmptyState");
const indexPath = document.getElementById("indexPath");
const pitchMethod = document.getElementById("pitchMethod");
const indexRate = document.getElementById("indexRate");
const protect = document.getElementById("protect");
const rmsMixRate = document.getElementById("rmsMixRate");
const filterRadius = document.getElementById("filterRadius");
const indexRateValue = document.getElementById("indexRateValue");
const protectValue = document.getElementById("protectValue");
const rmsMixRateValue = document.getElementById("rmsMixRateValue");
const filterRadiusValue = document.getElementById("filterRadiusValue");
const preprocessStrengthValue = document.getElementById("preprocessStrengthValue");

const generateGuideFileInput = document.getElementById("generateGuideFileInput");
const generateDropZone = document.getElementById("generateDropZone");
const generateFileList = document.getElementById("generateFileList");
const generateLyrics = document.getElementById("generateLyrics");
const generateGuideKey = document.getElementById("generateGuideKey");
const generateTargetKey = document.getElementById("generateTargetKey");
const generateGuideBpm = document.getElementById("generateGuideBpm");
const generateTargetBpm = document.getElementById("generateTargetBpm");
const generateQualityPreset = document.getElementById("generateQualityPreset");
const generateQualitySummary = document.getElementById("generateQualitySummary");
const generatePreprocessMode = document.getElementById("generatePreprocessMode");
const generatePreprocessModeSummary = document.getElementById("generatePreprocessModeSummary");
const generatePreprocessStrength = document.getElementById("generatePreprocessStrength");
const generatePreprocessStrengthValue = document.getElementById("generatePreprocessStrengthValue");
const startGenerateButton = document.getElementById("startGenerateButton");
const generateStatusCard = document.getElementById("generateStatusCard");
const generateStatusTitle = document.getElementById("generateStatusTitle");
const generateStatusMessage = document.getElementById("generateStatusMessage");
const generateProgressBar = document.getElementById("generateProgressBar");
const generateResultSummary = document.getElementById("generateResultSummary");
const generateResults = document.getElementById("generateResults");

const isolatorMode = document.getElementById("isolatorMode");
const isolatorModeSummary = document.getElementById("isolatorModeSummary");
const isolatorInputType = document.getElementById("isolatorInputType");
const isolatorInputTypeSummary = document.getElementById("isolatorInputTypeSummary");
const isolatorStrength = document.getElementById("isolatorStrength");
const isolatorStrengthValue = document.getElementById("isolatorStrengthValue");
const isolatorDeecho = document.getElementById("isolatorDeecho");
const isolatorWidthFocus = document.getElementById("isolatorWidthFocus");
const isolatorClarityPreserve = document.getElementById("isolatorClarityPreserve");
const isolatorClarityPreserveValue = document.getElementById("isolatorClarityPreserveValue");
const isolatorFileInput = document.getElementById("isolatorFileInput");
const isolatorDropZone = document.getElementById("isolatorDropZone");
const isolatorFileList = document.getElementById("isolatorFileList");
const startIsolatorButton = document.getElementById("startIsolatorButton");
const isolatorStatusCard = document.getElementById("isolatorStatusCard");
const isolatorStatusTitle = document.getElementById("isolatorStatusTitle");
const isolatorStatusMessage = document.getElementById("isolatorStatusMessage");
const isolatorProgressBar = document.getElementById("isolatorProgressBar");
const isolatorResultSummary = document.getElementById("isolatorResultSummary");
const isolatorResults = document.getElementById("isolatorResults");

const masteringResolution = document.getElementById("masteringResolution");
const masteringResolutionValue = document.getElementById("masteringResolutionValue");
const masteringSourceFileInput = document.getElementById("masteringSourceFileInput");
const masteringSourceDropZone = document.getElementById("masteringSourceDropZone");
const masteringSourceFileList = document.getElementById("masteringSourceFileList");
const masteringReferenceFileInput = document.getElementById("masteringReferenceFileInput");
const masteringReferenceDropZone = document.getElementById("masteringReferenceDropZone");
const masteringReferenceFileList = document.getElementById("masteringReferenceFileList");
const startMasteringButton = document.getElementById("startMasteringButton");
const masteringStatusCard = document.getElementById("masteringStatusCard");
const masteringStatusTitle = document.getElementById("masteringStatusTitle");
const masteringStatusMessage = document.getElementById("masteringStatusMessage");
const masteringProgressBar = document.getElementById("masteringProgressBar");
const masteringResultSummary = document.getElementById("masteringResultSummary");
const masteringProfileCard = document.getElementById("masteringProfileCard");
const masteringProfileMeta = document.getElementById("masteringProfileMeta");
const masteringProfileChart = document.getElementById("masteringProfileChart");
const masteringResults = document.getElementById("masteringResults");

const optimizeStrength = document.getElementById("optimizeStrength");
const optimizeStrengthValue = document.getElementById("optimizeStrengthValue");
const optimizeLyrics = document.getElementById("optimizeLyrics");
const optimizeFileInput = document.getElementById("optimizeFileInput");
const optimizeDropZone = document.getElementById("optimizeDropZone");
const optimizeFileList = document.getElementById("optimizeFileList");
const startOptimizeButton = document.getElementById("startOptimizeButton");
const optimizeStatusCard = document.getElementById("optimizeStatusCard");
const optimizeStatusTitle = document.getElementById("optimizeStatusTitle");
const optimizeStatusMessage = document.getElementById("optimizeStatusMessage");
const optimizeProgressBar = document.getElementById("optimizeProgressBar");
const optimizeResultSummary = document.getElementById("optimizeResultSummary");
const optimizeResults = document.getElementById("optimizeResults");

const albumName = document.getElementById("albumName");
const createAlbumButton = document.getElementById("createAlbumButton");
const refreshAlbumsButton = document.getElementById("refreshAlbumsButton");
const albumProjectSelect = document.getElementById("albumProjectSelect");
const albumBulkFileInput = document.getElementById("albumBulkFileInput");
const albumBulkDropZone = document.getElementById("albumBulkDropZone");
const albumBulkFileList = document.getElementById("albumBulkFileList");
const albumBulkUploadButton = document.getElementById("albumBulkUploadButton");
const albumSongsList = document.getElementById("albumSongsList");
const albumStatusCard = document.getElementById("albumStatusCard");
const albumStatusTitle = document.getElementById("albumStatusTitle");
const albumStatusMessage = document.getElementById("albumStatusMessage");
const albumProgressBar = document.getElementById("albumProgressBar");
const albumResultSummary = document.getElementById("albumResultSummary");
const albumMixCard = document.getElementById("albumMixCard");
const albumPlayMixButton = document.getElementById("albumPlayMixButton");
const albumNowPlayingTitle = document.getElementById("albumNowPlayingTitle");
const albumNowPlayingMeta = document.getElementById("albumNowPlayingMeta");
const albumPreviewPlayer = document.getElementById("albumPreviewPlayer");
const albumPreviewDownload = document.getElementById("albumPreviewDownload");
const albumLogList = document.getElementById("albumLogList");

const apiComposeEndpointUrl = document.getElementById("apiComposeEndpointUrl");
const apiComposeAuthHeader = document.getElementById("apiComposeAuthHeader");
const apiComposeApiKey = document.getElementById("apiComposeApiKey");
const apiComposeMidiFileInput = document.getElementById("apiComposeMidiFileInput");
const apiComposeMidiDropZone = document.getElementById("apiComposeMidiDropZone");
const apiComposeMidiFileList = document.getElementById("apiComposeMidiFileList");
const apiComposeBeatFileInput = document.getElementById("apiComposeBeatFileInput");
const apiComposeBeatDropZone = document.getElementById("apiComposeBeatDropZone");
const apiComposeBeatFileList = document.getElementById("apiComposeBeatFileList");
const apiComposeLyrics = document.getElementById("apiComposeLyrics");
const apiComposeExtraJson = document.getElementById("apiComposeExtraJson");
const checkApiComposeHealthButton = document.getElementById("checkApiComposeHealthButton");
const startApiComposeButton = document.getElementById("startApiComposeButton");
const apiComposeStatusCard = document.getElementById("apiComposeStatusCard");
const apiComposeStatusTitle = document.getElementById("apiComposeStatusTitle");
const apiComposeStatusMessage = document.getElementById("apiComposeStatusMessage");
const apiComposeProgressBar = document.getElementById("apiComposeProgressBar");
const apiComposeResultSummary = document.getElementById("apiComposeResultSummary");
const apiComposeResults = document.getElementById("apiComposeResults");
const apiComposeResponse = document.getElementById("apiComposeResponse");

const touchUpReferenceWord = document.getElementById("touchUpReferenceWord");
const touchUpSourceWord = document.getElementById("touchUpSourceWord");
const touchUpMode = document.getElementById("touchUpMode");
const touchUpStrength = document.getElementById("touchUpStrength");
const touchUpStrengthValue = document.getElementById("touchUpStrengthValue");
const touchUpMaxWords = document.getElementById("touchUpMaxWords");
const touchUpReferenceFileInput = document.getElementById("touchUpReferenceFileInput");
const touchUpReferenceDropZone = document.getElementById("touchUpReferenceDropZone");
const touchUpReferenceFileList = document.getElementById("touchUpReferenceFileList");
const touchUpSourceFileInput = document.getElementById("touchUpSourceFileInput");
const touchUpSourceDropZone = document.getElementById("touchUpSourceDropZone");
const touchUpSourceFileList = document.getElementById("touchUpSourceFileList");
const startTouchUpButton = document.getElementById("startTouchUpButton");
const stopTouchUpButton = document.getElementById("stopTouchUpButton");
const touchUpStatusCard = document.getElementById("touchUpStatusCard");
const touchUpStatusTitle = document.getElementById("touchUpStatusTitle");
const touchUpStatusMessage = document.getElementById("touchUpStatusMessage");
const touchUpProgressBar = document.getElementById("touchUpProgressBar");
const touchUpResultSummary = document.getElementById("touchUpResultSummary");
const touchUpResults = document.getElementById("touchUpResults");

const detagVoiceSelect = document.getElementById("detagVoiceSelect");
const detagStrength = document.getElementById("detagStrength");
const detagStrengthValue = document.getElementById("detagStrengthValue");
const detagFileInput = document.getElementById("detagFileInput");
const detagDropZone = document.getElementById("detagDropZone");
const detagFileList = document.getElementById("detagFileList");
const detagEmptyState = document.getElementById("detagEmptyState");
const startDetagButton = document.getElementById("startDetagButton");
const detagStatusCard = document.getElementById("detagStatusCard");
const detagStatusTitle = document.getElementById("detagStatusTitle");
const detagStatusMessage = document.getElementById("detagStatusMessage");
const detagProgressBar = document.getElementById("detagProgressBar");
const detagResultSummary = document.getElementById("detagResultSummary");
const detagResults = document.getElementById("detagResults");

const trainingName = document.getElementById("trainingName");
const trainingVersion = document.getElementById("trainingVersion");
const trainingSampleRate = document.getElementById("trainingSampleRate");
const trainingF0Method = document.getElementById("trainingF0Method");
const trainingOutputMode = document.getElementById("trainingOutputMode");
const trainingModeSummary = document.getElementById("trainingModeSummary");
const trainingPackageDownloadSelect = document.getElementById("trainingPackageDownloadSelect");
const downloadTrainingPackageButton = document.getElementById("downloadTrainingPackageButton");
const trainingPackageDownloadSummary = document.getElementById("trainingPackageDownloadSummary");
const trainingResumePackageSelect = document.getElementById("trainingResumePackageSelect");
const trainingStartPhase = document.getElementById("trainingStartPhase");
const trainingResumeSummary = document.getElementById("trainingResumeSummary");
const trainingEpochMode = document.getElementById("trainingEpochMode");
const trainingEpochModeSummary = document.getElementById("trainingEpochModeSummary");
const trainingEpochs = document.getElementById("trainingEpochs");
const trainingCurriculumSummary = document.getElementById("trainingCurriculumSummary");
const trainingSaveEvery = document.getElementById("trainingSaveEvery");
const trainingBatchSize = document.getElementById("trainingBatchSize");
const trainingCrepeHopLength = document.getElementById("trainingCrepeHopLength");
const trainingFileInput = document.getElementById("trainingFileInput");
const trainingDropZone = document.getElementById("trainingDropZone");
const trainingFileList = document.getElementById("trainingFileList");
const trainingPlanFileInput = document.getElementById("trainingPlanFileInput");
const trainingPlanDropZone = document.getElementById("trainingPlanDropZone");
const trainingPlanFileList = document.getElementById("trainingPlanFileList");
const trainingTranscriptFileInput = document.getElementById("trainingTranscriptFileInput");
const trainingTranscriptDropZone = document.getElementById("trainingTranscriptDropZone");
const trainingTranscriptFileList = document.getElementById("trainingTranscriptFileList");
const trainingWarning = document.getElementById("trainingWarning");
const startTrainingButton = document.getElementById("startTrainingButton");
const stopTrainingButton = document.getElementById("stopTrainingButton");
const trainingStatusCard = document.getElementById("trainingStatusCard");
const trainingStatusTitle = document.getElementById("trainingStatusTitle");
const trainingStatusMessage = document.getElementById("trainingStatusMessage");
const trainingProgressBar = document.getElementById("trainingProgressBar");
const trainingResultSummary = document.getElementById("trainingResultSummary");
const trainingCheckpointPreview = document.getElementById("trainingCheckpointPreview");
const trainingCheckpointPreviewMeta = document.getElementById("trainingCheckpointPreviewMeta");
const trainingCheckpointPreviewPlayer = document.getElementById("trainingCheckpointPreviewPlayer");
const trainingLog = document.getElementById("trainingLog");

function bindIfPresent(element, eventName, handler) {
  if (element) {
    element.addEventListener(eventName, handler);
  }
}

function setPanelStatus(card, titleEl, messageEl, barEl, mode, title, message, percent = 0) {
  if (!card || !titleEl || !messageEl || !barEl) {
    return;
  }
  card.className = `status-card ${mode}`;
  titleEl.textContent = title;
  messageEl.textContent = message;
  barEl.style.width = `${percent}%`;
}

function setStatus(mode, title, message, percent = 0) {
  setPanelStatus(statusCard, statusTitle, statusMessage, progressBar, mode, title, message, percent);
}

function setGenerateStatus(mode, title, message, percent = 0) {
  setPanelStatus(
    generateStatusCard,
    generateStatusTitle,
    generateStatusMessage,
    generateProgressBar,
    mode,
    title,
    message,
    percent,
  );
}

function setPreviewStatus(mode, title, message, percent = 0) {
  setPanelStatus(
    previewStatusCard,
    previewStatusTitle,
    previewStatusMessage,
    previewProgressBar,
    mode,
    title,
    message,
    percent,
  );
}

function setIsolatorStatus(mode, title, message, percent = 0) {
  setPanelStatus(
    isolatorStatusCard,
    isolatorStatusTitle,
    isolatorStatusMessage,
    isolatorProgressBar,
    mode,
    title,
    message,
    percent,
  );
}

function setMasteringStatus(mode, title, message, percent = 0) {
  setPanelStatus(
    masteringStatusCard,
    masteringStatusTitle,
    masteringStatusMessage,
    masteringProgressBar,
    mode,
    title,
    message,
    percent,
  );
}

function setOptimizeStatus(mode, title, message, percent = 0) {
  setPanelStatus(
    optimizeStatusCard,
    optimizeStatusTitle,
    optimizeStatusMessage,
    optimizeProgressBar,
    mode,
    title,
    message,
    percent,
  );
}

function setAlbumStatus(mode, title, message, percent = 0) {
  setPanelStatus(
    albumStatusCard,
    albumStatusTitle,
    albumStatusMessage,
    albumProgressBar,
    mode,
    title,
    message,
    percent,
  );
}

function setApiComposeStatus(mode, title, message, percent = 0) {
  setPanelStatus(
    apiComposeStatusCard,
    apiComposeStatusTitle,
    apiComposeStatusMessage,
    apiComposeProgressBar,
    mode,
    title,
    message,
    percent,
  );
}

function setTouchUpStatus(mode, title, message, percent = 0) {
  setPanelStatus(
    touchUpStatusCard,
    touchUpStatusTitle,
    touchUpStatusMessage,
    touchUpProgressBar,
    mode,
    title,
    message,
    percent,
  );
}

function setTrainingStatus(mode, title, message, percent = 0) {
  setPanelStatus(
    trainingStatusCard,
    trainingStatusTitle,
    trainingStatusMessage,
    trainingProgressBar,
    mode,
    title,
    message,
    percent,
  );
}

function setDetagStatus(mode, title, message, percent = 0) {
  setPanelStatus(
    detagStatusCard,
    detagStatusTitle,
    detagStatusMessage,
    detagProgressBar,
    mode,
    title,
    message,
    percent,
  );
}

function getPreprocessPipelineMeta(pipelineId) {
  return (state.preprocessPipelines || []).find((entry) => entry.id === pipelineId) || null;
}

function getPreprocessLabel(pipelineId) {
  if (!pipelineId || pipelineId === "off") {
    return "Off";
  }
  return getPreprocessPipelineMeta(pipelineId)?.label || pipelineId;
}

function updatePreprocessModeSummary() {
  if (!preprocessModeSummary) {
    return;
  }
  preprocessModeSummary.textContent =
    "Pre-conversion isolation is disabled in Lead Builder. Upload your already-isolated vocal and we convert it directly.";
}

function updateGeneratePreprocessModeSummary() {
  if (!generatePreprocessModeSummary) {
    return;
  }
  const selected = getPreprocessPipelineMeta(generatePreprocessMode?.value || "");
  if (!selected) {
    generatePreprocessModeSummary.textContent =
      "Clean the reference vocal before conversion so the repair stage starts from a better take.";
    return;
  }
  generatePreprocessModeSummary.textContent = selected.description || selected.label || "";
}

function updateOutputModeSummary() {
  if (!outputModeSummary || !outputMode) {
    return;
  }
  if (outputMode.value === "blend") {
    outputModeSummary.textContent =
      "Blend mode renders both selected target packages, then merges them into one final lead using the balance slider below.";
    return;
  }
  outputModeSummary.textContent =
    "Single voice uses one target package for the entire rebuilt lead.";
}

function updateOutputModeUI() {
  const blendActive = outputMode && outputMode.value === "blend";
  if (secondaryModelField) {
    secondaryModelField.classList.toggle("hidden", !blendActive);
  }
  if (blendPercentageField) {
    blendPercentageField.classList.toggle("hidden", !blendActive);
  }
  updateOutputModeSummary();
}

function updateSliderLabels() {
  if (indexRateValue && indexRate) indexRateValue.textContent = `${indexRate.value}%`;
  if (protectValue && protect) protectValue.textContent = `${protect.value}%`;
  if (rmsMixRateValue && rmsMixRate) rmsMixRateValue.textContent = `${rmsMixRate.value}%`;
  if (filterRadiusValue && filterRadius) filterRadiusValue.textContent = filterRadius.value;
  if (preprocessStrengthValue && preprocessStrength) preprocessStrengthValue.textContent = preprocessStrength.value;
  if (blendPercentageValue && blendPercentage) {
    blendPercentageValue.textContent = `${blendPercentage.value}%`;
  }
  if (secondaryBlendPercentageValue && blendPercentage) {
    secondaryBlendPercentageValue.textContent = `${100 - Number(blendPercentage.value || 0)}%`;
  }
  if (generatePreprocessStrengthValue && generatePreprocessStrength) {
    generatePreprocessStrengthValue.textContent = generatePreprocessStrength.value;
  }
  if (isolatorStrengthValue && isolatorStrength) isolatorStrengthValue.textContent = isolatorStrength.value;
  if (isolatorClarityPreserveValue && isolatorClarityPreserve) {
    isolatorClarityPreserveValue.textContent = `${isolatorClarityPreserve.value}%`;
  }
  if (masteringResolutionValue && masteringResolution) {
    masteringResolutionValue.textContent = masteringResolution.value;
  }
  if (optimizeStrengthValue && optimizeStrength) {
    optimizeStrengthValue.textContent = `${optimizeStrength.value} dB`;
  }
  if (touchUpStrengthValue && touchUpStrength) touchUpStrengthValue.textContent = touchUpStrength.value;
  if (detagStrengthValue && detagStrength) detagStrengthValue.textContent = detagStrength.value;
}

function syncTrainingRunModeUI() {
  if (trainingEpochMode) {
    trainingEpochMode.value = "fixed";
  }
  const modeConfig = getTrainingModeConfig(trainingOutputMode?.value || "classic-rvc-support");
  if (trainingModeSummary) {
    trainingModeSummary.textContent = modeConfig.summary;
  }
  if (trainingEpochModeSummary) {
    trainingEpochModeSummary.textContent = modeConfig.requiresPairs
      ? `${modeConfig.label} now uses one real epoch count. Legacy warm-up, bridge, refine, matching, and worker knobs are disabled on this screen.`
      : `${modeConfig.label} now uses one real epoch count. BASE clips stay as the main truth, and the fixed SUNO audition clip is only used for checkpoint listening.`;
  }
  renderTrainingFiles();
  updateTrainingCurriculumSummary();
}

function clampTrainingEpochInput(input, fallback) {
  if (!input) {
    return fallback;
  }
  const parsed = Number.parseInt(input.value || "", 10);
  const nextValue = Number.isFinite(parsed) ? Math.max(1, Math.min(parsed, 30000)) : fallback;
  input.value = String(nextValue);
  return nextValue;
}

function updateTrainingCurriculumSummary() {
  const totalEpochs = clampTrainingEpochInput(trainingEpochs, 600);
  if (!trainingCurriculumSummary) {
    return;
  }
  const modeConfig = getTrainingModeConfig(trainingOutputMode?.value || "classic-rvc-support");
  trainingCurriculumSummary.textContent = modeConfig.requiresPairs
    ? `The ${modeConfig.label.toLowerCase()} run will train for ${totalEpochs} epochs and then stop. Save frequency, batch size, and Crepe hop length are the only remaining advanced controls here.`
    : `The ${modeConfig.label.toLowerCase()} run will train for ${totalEpochs} epochs and then stop. BASE clips stay dominant, optional TARGET clips add more true voice coverage, and the same SUNO audition clip is rendered again at every saved checkpoint.`;
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatRelativeTime(isoText) {
  if (!isoText) return "";
  const parsed = new Date(isoText);
  if (Number.isNaN(parsed.getTime())) return "";
  const diffMs = Date.now() - parsed.getTime();
  const diffSeconds = Math.max(1, Math.floor(diffMs / 1000));
  if (diffSeconds < 60) return "just now";
  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes} minute${diffMinutes === 1 ? "" : "s"} ago`;
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours} hour${diffHours === 1 ? "" : "s"} ago`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 30) return `${diffDays} day${diffDays === 1 ? "" : "s"} ago`;
  const diffMonths = Math.floor(diffDays / 30);
  if (diffMonths < 12) return `${diffMonths} month${diffMonths === 1 ? "" : "s"} ago`;
  const diffYears = Math.floor(diffDays / 365);
  return `${diffYears} year${diffYears === 1 ? "" : "s"} ago`;
}

function formatDuration(seconds) {
  const totalSeconds = Math.max(0, Math.floor(Number(seconds || 0)));
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

function playAlbumSource(url, title, meta = "", autoPlay = true) {
  if (!url || !albumPreviewPlayer) {
    return;
  }
  albumPreviewPlayer.pause();
  albumPreviewPlayer.currentTime = 0;
  setManagedAudioMeta(albumPreviewPlayer, title, meta);
  albumPreviewPlayer.src = url;
  albumNowPlayingTitle.textContent = title || "Now playing";
  albumNowPlayingMeta.textContent = meta || "";
  albumPreviewPlayer.load();
  if (autoPlay) {
    albumPreviewPlayer.play().catch(() => {});
  }
}

function isTypingField(element) {
  if (!element || !(element instanceof HTMLElement)) {
    return false;
  }
  if (element.isContentEditable) {
    return true;
  }
  if (element.tagName === "TEXTAREA") {
    return true;
  }
  if (element.tagName === "INPUT") {
    const type = (element.getAttribute("type") || "text").toLowerCase();
    return !["range", "checkbox", "radio", "button", "submit", "reset", "color", "file"].includes(type);
  }
  return element.tagName === "SELECT";
}

function toggleManagedAudioPlayback() {
  const candidates = [activeManagedAudio, albumPreviewPlayer, previewPlayer].filter(Boolean);
  const audioEl = candidates.find((item) => item && item.src) || candidates[0] || null;
  if (!audioEl) {
    return false;
  }
  if (audioEl.paused) {
    audioEl.play().catch(() => {});
  } else {
    audioEl.pause();
  }
  return true;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function buildManagedAudioMarkup(url, title, subtitle = "") {
  return `<audio class="managed-audio-source" preload="metadata" src="${escapeHtml(url)}" data-player-title="${escapeHtml(title)}" data-player-subtitle="${escapeHtml(subtitle)}"></audio>`;
}

function appendDirectConversionResultCard(container, result) {
  if (!container || !result) {
    return;
  }
  const card = document.createElement("div");
  card.className = "result-row";

  const name = document.createElement("div");
  name.className = "result-name";
  name.textContent = result.download_name || result.name || "converted.wav";
  card.appendChild(name);

  const meta = document.createElement("div");
  meta.className = "result-meta";
  meta.textContent = `Sample rate: ${Number(result.sample_rate || 0)} Hz. Timing: npy ${Number(result.timings?.npy || 0).toFixed(2)}s, f0 ${Number(result.timings?.f0 || 0).toFixed(2)}s, infer ${Number(result.timings?.infer || 0).toFixed(2)}s.`;
  card.appendChild(meta);

  if (result.url) {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = result.url;
    audio.className = "direct-audio-player";
    card.appendChild(audio);

    const links = document.createElement("div");
    links.className = "result-links";
    const link = document.createElement("a");
    link.className = "result-link";
    link.href = result.url;
    link.download = result.download_name || result.name || "converted.wav";
    link.textContent = "Download file";
    links.appendChild(link);
    card.appendChild(links);
  } else {
    const missing = document.createElement("div");
    missing.className = "result-meta";
    missing.textContent = "Output file is missing a downloadable URL.";
    card.appendChild(missing);
  }

  container.appendChild(card);
}

function normalizeDirectConversionResult(result) {
  if (!result) {
    return null;
  }
  const safeTimings = result.timings && typeof result.timings === "object" ? result.timings : {};
  return {
    name: result.name || result.download_name || "converted.wav",
    url: String(result.url || result.result_url || "").trim(),
    download_name: result.download_name || result.name || "converted.wav",
    sample_rate: Number(result.sample_rate || 0),
    timings: safeTimings,
  };
}

function buildDirectConversionDownloadMarkup(result, label = "Download file") {
  if (!result?.url) {
    return "";
  }
  const href = escapeHtml(result.url);
  const downloadName = escapeHtml(result.download_name || result.name || "converted.wav");
  return `<a class="result-link" href="${href}" download="${downloadName}">${escapeHtml(label)}</a>`;
}

function showDirectConversionPreview(result) {
  if (!previewMeta || !previewPlayer) {
    return;
  }
  if (!result?.url) {
    resetPreviewState(
      "Preview disabled",
      "Lead Builder now runs only the full direct-conversion path instead of the old 5 second preview.",
    );
    return;
  }

  const fileLabel = result.download_name || result.name || "converted.wav";
  const timingSummary = [
    `npy ${Number(result.timings?.npy || 0).toFixed(2)}s`,
    `f0 ${Number(result.timings?.f0 || 0).toFixed(2)}s`,
    `infer ${Number(result.timings?.infer || 0).toFixed(2)}s`,
  ].join(" | ");
  const subtitle = [
    Number(result.sample_rate || 0) > 0 ? `${Number(result.sample_rate)} Hz` : "",
    timingSummary,
  ].filter(Boolean).join(" | ");

  previewPlayer.pause();
  previewPlayer.currentTime = 0;
  setManagedAudioMeta(previewPlayer, fileLabel, subtitle || "Latest direct conversion");
  previewPlayer.src = result.url;
  previewPlayer.classList.remove("hidden");
  previewPlayer.load();
  setManagedAudioVisible(previewPlayer, true);

  previewMeta.classList.remove("hidden");
  previewMeta.innerHTML = `
    <strong>Latest conversion ready</strong>
    <p>${escapeHtml(fileLabel)} is loaded into the player below.</p>
    <div class="result-links">
      ${buildDirectConversionDownloadMarkup(result, "Download latest converted file")}
    </div>
  `;
  setPreviewStatus("completed", "Latest conversion ready", "Play or download the finished vocal below.", 100);
}

function renderDirectConversionCompletion(job, renderableResults) {
  const normalizedResults = (renderableResults || [])
    .map((result) => normalizeDirectConversionResult(result))
    .filter(Boolean);
  const primaryResult = normalizedResults.find((result) => result.url) || normalizedResults[0] || null;

  resultSummary.classList.toggle("hidden", !normalizedResults.length && !job.zip_url);
  resultSummary.innerHTML = `
    <strong>${state.currentConversionModelSystem || "Direct conversion"} finished</strong>
    <p>Voice: ${escapeHtml(state.currentConversionModelLabel || modelSelect?.value || "unknown model")}.</p>
    <p>Converted file${normalizedResults.length === 1 ? "" : "s"}: ${normalizedResults.length}. This render used only the trained \`.pth\` converter with no lyric repair or post-processing stage.</p>
    <div class="result-links">
      ${primaryResult?.url ? buildDirectConversionDownloadMarkup(primaryResult, "Download latest converted file") : ""}
      ${job.zip_url ? `<a class="zip-link" href="${escapeHtml(job.zip_url)}" download>Download converted package</a>` : ""}
    </div>
  `;

  results.replaceChildren();
  normalizedResults.forEach((result) => {
    appendDirectConversionResultCard(results, result);
  });

  showDirectConversionPreview(primaryResult);
}

function renderDirectConversionFallback(job, renderableResults, error) {
  const normalizedResults = (renderableResults || [])
    .map((result) => normalizeDirectConversionResult(result))
    .filter(Boolean);
  const primaryResult = normalizedResults.find((result) => result.url) || normalizedResults[0] || null;
  console.error("Direct conversion result render failed.", error, job);

  resultSummary.classList.remove("hidden");
  resultSummary.innerHTML = `
    <strong>${state.currentConversionModelSystem || "Direct conversion"} finished</strong>
    <p>The rich result card failed to render, so the page is showing the raw output link instead.</p>
    <p>${escapeHtml(error?.message || String(error || "Unknown browser render error"))}</p>
    <div class="result-links">
      ${primaryResult?.url ? buildDirectConversionDownloadMarkup(primaryResult, "Download raw converted file") : ""}
      ${job?.zip_url ? `<a class="zip-link" href="${escapeHtml(job.zip_url)}" download>Download converted package</a>` : ""}
    </div>
  `;

  results.replaceChildren();
  if (primaryResult) {
    appendDirectConversionResultCard(results, primaryResult);
  }

  showDirectConversionPreview(primaryResult);
}

let activeManagedAudio = null;

function formatPlayerTime(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) {
    return "0:00";
  }
  return formatDuration(value);
}

function updateManagedAudioShell(audioEl) {
  const shell = audioEl?._managedPlayerShell;
  if (!shell) {
    return;
  }
  shell.classList.toggle("is-active", !audioEl.paused);
  const playButton = shell.querySelector(".managed-audio-toggle");
  const seek = shell.querySelector(".managed-audio-seek");
  const current = shell.querySelector(".managed-audio-current");
  const duration = shell.querySelector(".managed-audio-duration");
  const title = shell.querySelector(".managed-audio-title");
  const subtitle = shell.querySelector(".managed-audio-subtitle");
  if (title) {
    title.textContent = audioEl.dataset.playerTitle || "Audio";
  }
  if (subtitle) {
    subtitle.textContent = audioEl.dataset.playerSubtitle || "";
  }
  if (playButton) {
    playButton.textContent = audioEl.paused ? "Play" : "Pause";
  }
  if (seek) {
    const max = Number.isFinite(audioEl.duration) && audioEl.duration > 0 ? audioEl.duration : 0;
    seek.max = String(max);
    if (!audioEl._managedSeeking) {
      seek.value = String(Math.min(audioEl.currentTime || 0, max));
    }
  }
  if (current) {
    current.textContent = formatPlayerTime(audioEl.currentTime || 0);
  }
  if (duration) {
    duration.textContent = formatPlayerTime(audioEl.duration || 0);
  }
}

function setManagedAudioMeta(audioEl, title, subtitle = "") {
  if (!audioEl) {
    return;
  }
  audioEl.dataset.playerTitle = title || "";
  audioEl.dataset.playerSubtitle = subtitle || "";
  updateManagedAudioShell(audioEl);
}

function setManagedAudioVisible(audioEl, visible) {
  if (!audioEl) {
    return;
  }
  const shell = audioEl._managedPlayerShell;
  if (shell) {
    shell.classList.toggle("hidden", !visible);
  }
}

function ensureManagedAudioPlayer(audioEl) {
  if (!audioEl) {
    return null;
  }
  if (audioEl.dataset.managedReady === "1" && audioEl._managedPlayerShell) {
    updateManagedAudioShell(audioEl);
    return audioEl._managedPlayerShell;
  }

  audioEl.dataset.managedReady = "1";
  audioEl.classList.add("native-audio-hidden");
  audioEl.removeAttribute("controls");
  audioEl.preload = "metadata";

  const shell = document.createElement("div");
  shell.className = "managed-audio-player";
  shell.innerHTML = `
    <button class="managed-audio-toggle" type="button">Play</button>
    <div class="managed-audio-copy">
      <div class="managed-audio-title"></div>
      <div class="managed-audio-subtitle"></div>
    </div>
    <div class="managed-audio-time managed-audio-current">0:00</div>
    <input class="managed-audio-seek" type="range" min="0" max="0" value="0" step="0.01">
    <div class="managed-audio-time managed-audio-duration">0:00</div>
  `;
  audioEl.insertAdjacentElement("afterend", shell);
  audioEl._managedPlayerShell = shell;

  const playButton = shell.querySelector(".managed-audio-toggle");
  const seek = shell.querySelector(".managed-audio-seek");

  playButton?.addEventListener("click", () => {
    if (audioEl.paused) {
      audioEl.play().catch(() => {});
    } else {
      audioEl.pause();
    }
  });

  seek?.addEventListener("pointerdown", () => {
    audioEl._managedSeeking = true;
  });
  seek?.addEventListener("input", () => {
    audioEl._managedSeeking = true;
    audioEl.currentTime = Number(seek.value || 0);
    updateManagedAudioShell(audioEl);
  });
  seek?.addEventListener("change", () => {
    audioEl.currentTime = Number(seek.value || 0);
    audioEl._managedSeeking = false;
    updateManagedAudioShell(audioEl);
  });
  seek?.addEventListener("pointerup", () => {
    audioEl.currentTime = Number(seek.value || 0);
    audioEl._managedSeeking = false;
    updateManagedAudioShell(audioEl);
  });

  audioEl.addEventListener("loadedmetadata", () => updateManagedAudioShell(audioEl));
  audioEl.addEventListener("durationchange", () => updateManagedAudioShell(audioEl));
  audioEl.addEventListener("seeked", () => updateManagedAudioShell(audioEl));
  audioEl.addEventListener("timeupdate", () => updateManagedAudioShell(audioEl));
  audioEl.addEventListener("play", () => {
    if (activeManagedAudio && activeManagedAudio !== audioEl) {
      activeManagedAudio.pause();
    }
    activeManagedAudio = audioEl;
    updateManagedAudioShell(audioEl);
  });
  audioEl.addEventListener("pause", () => updateManagedAudioShell(audioEl));
  audioEl.addEventListener("ended", () => updateManagedAudioShell(audioEl));

  if (audioEl.src) {
    try {
      audioEl.load();
    } catch {
      // Ignore load issues for broken or cross-origin sources.
    }
  }

  updateManagedAudioShell(audioEl);
  return shell;
}

function hydrateManagedAudio(root = document) {
  root.querySelectorAll("audio.managed-audio-source").forEach((audioEl) => {
    ensureManagedAudioPlayer(audioEl);
  });
}

async function reorderAlbumTracks(songIndices) {
  if (!state.currentAlbumProjectId || !Array.isArray(songIndices) || !songIndices.length) {
    return;
  }
  const response = await fetch(`/api/albums/projects/${state.currentAlbumProjectId}/reorder`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ song_indices: songIndices }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Could not save track order.");
  }
  renderAlbumProject(payload.project || null);
  await loadAlbumProjects(false);
  renderAlbumProjectSelector();
}

function wireAlbumTrackReorder() {
  const rows = Array.from(albumSongsList.querySelectorAll(".album-track-row[data-song-index]"));
  const handles = Array.from(albumSongsList.querySelectorAll(".album-drag-handle"));
  handles.forEach((handle) => {
    const row = handle.closest(".album-track-row");
    if (!row) {
      return;
    }

    handle.addEventListener("dragstart", (event) => {
      state.albumDragSongIndex = row.dataset.songIndex || null;
      row.classList.add("dragging");
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", row.dataset.songIndex || "");
      }
    });

    handle.addEventListener("dragend", () => {
      state.albumDragSongIndex = null;
      rows.forEach((item) => item.classList.remove("dragging", "drag-over-top", "drag-over-bottom"));
    });
  });

  rows.forEach((row) => {
    row.addEventListener("dragover", (event) => {
      event.preventDefault();
      const rect = row.getBoundingClientRect();
      const placeAfter = event.clientY > rect.top + rect.height / 2;
      row.classList.toggle("drag-over-top", !placeAfter);
      row.classList.toggle("drag-over-bottom", placeAfter);
    });

    row.addEventListener("dragleave", () => {
      row.classList.remove("drag-over-top", "drag-over-bottom");
    });

    row.addEventListener("drop", async (event) => {
      event.preventDefault();
      const dragged = state.albumDragSongIndex || event.dataTransfer?.getData("text/plain") || "";
      const target = row.dataset.songIndex || "";
      rows.forEach((item) => item.classList.remove("drag-over-top", "drag-over-bottom"));
      if (!dragged || !target || dragged === target) {
        return;
      }

      const currentOrder = rows.map((item) => Number(item.dataset.songIndex || 0)).filter(Boolean);
      const sourceIndex = currentOrder.indexOf(Number(dragged));
      const targetIndex = currentOrder.indexOf(Number(target));
      if (sourceIndex < 0 || targetIndex < 0) {
        return;
      }

      const rect = row.getBoundingClientRect();
      const placeAfter = event.clientY > rect.top + rect.height / 2;
      const reordered = [...currentOrder];
      const [moved] = reordered.splice(sourceIndex, 1);
      let insertIndex = targetIndex;
      if (sourceIndex < targetIndex) {
        insertIndex -= 1;
      }
      if (placeAfter) {
        insertIndex += 1;
      }
      reordered.splice(Math.max(0, insertIndex), 0, moved);

      setAlbumStatus("running", "Saving order", "Updating album sequence...", 55);
      try {
        await reorderAlbumTracks(reordered);
        setAlbumStatus("completed", "Order saved", "Album track order updated.", 100);
      } catch (error) {
        setAlbumStatus("failed", "Could not reorder", error.message || String(error), 0);
      }
    });
  });
}

function extractDroppedFiles(event) {
  const collected = [];
  const items = Array.from(event.dataTransfer?.items || []);
  for (const item of items) {
    if (item.kind !== "file") {
      continue;
    }
    const file = item.getAsFile();
    if (file) {
      collected.push(file);
    }
  }

  if (collected.length) {
    return collected;
  }

  return Array.from(event.dataTransfer?.files || []);
}

function setActiveTab(tabName) {
  const convertActive = tabName === "convert";
  const generateActive = tabName === "generate";
  const isolatorActive = tabName === "isolator";
  const masteringActive = tabName === "mastering";
  const optimizeActive = tabName === "optimize";
  const albumsActive = tabName === "albums";
  const apiComposeActive = tabName === "api-compose";
  const touchUpActive = tabName === "touchup";
  const detagActive = tabName === "detag";
  const trainingActive = tabName === "training";
  if (convertTab) {
    convertTab.classList.toggle("hidden", !convertActive);
    convertTab.classList.toggle("active", convertActive);
  }
  if (generateTab) {
    generateTab.classList.toggle("hidden", !generateActive);
    generateTab.classList.toggle("active", generateActive);
  }
  if (isolatorTab) {
    isolatorTab.classList.toggle("hidden", !isolatorActive);
    isolatorTab.classList.toggle("active", isolatorActive);
  }
  if (masteringTab) {
    masteringTab.classList.toggle("hidden", !masteringActive);
    masteringTab.classList.toggle("active", masteringActive);
  }
  if (optimizeTab) {
    optimizeTab.classList.toggle("hidden", !optimizeActive);
    optimizeTab.classList.toggle("active", optimizeActive);
  }
  if (albumsTab) {
    albumsTab.classList.toggle("hidden", !albumsActive);
    albumsTab.classList.toggle("active", albumsActive);
  }
  if (apiComposeTab) {
    apiComposeTab.classList.toggle("hidden", !apiComposeActive);
    apiComposeTab.classList.toggle("active", apiComposeActive);
  }
  if (touchUpTab) {
    touchUpTab.classList.toggle("hidden", !touchUpActive);
    touchUpTab.classList.toggle("active", touchUpActive);
  }
  if (detagTab) {
    detagTab.classList.toggle("hidden", !detagActive);
    detagTab.classList.toggle("active", detagActive);
  }
  if (trainingTab) {
    trainingTab.classList.toggle("hidden", !trainingActive);
    trainingTab.classList.toggle("active", trainingActive);
  }
  [
    [showConvertTabButton, convertActive],
    [showGenerateTabButton, generateActive],
    [showIsolatorTabButton, isolatorActive],
    [showMasteringTabButton, masteringActive],
    [showOptimizeTabButton, optimizeActive],
    [showAlbumsTabButton, albumsActive],
    [showApiComposeTabButton, apiComposeActive],
    [showTouchUpTabButton, touchUpActive],
    [showDetagTabButton, detagActive],
    [showTrainingTabButton, trainingActive],
    [railConvertTabButton, convertActive],
    [railIsolatorTabButton, isolatorActive],
    [railMasteringTabButton, masteringActive],
    [railTrainingTabButton, trainingActive],
  ].forEach(([button, active]) => {
    if (button) {
      button.classList.toggle("active", active);
    }
  });
}

function renderFiles() {
  if (!state.files.length) {
    fileList.className = "file-list empty-list";
    fileList.innerHTML = "<p>No song selected yet.</p>";
    resetPreviewState("Preview disabled", "Lead Builder now runs only the full direct-conversion path.");
    return;
  }

  fileList.className = "file-list";
  fileList.innerHTML = "";
  state.files.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `
      <div>
        <div class="file-name">${file.name}</div>
        <div class="file-meta">${formatBytes(file.size)}</div>
      </div>
      <button class="remove-file" type="button" data-index="${index}">Remove</button>
    `;
    fileList.appendChild(row);
  });

  fileList.querySelectorAll(".remove-file").forEach((button) => {
    button.addEventListener("click", () => {
      const removedIndex = Number(button.dataset.index);
      state.files.splice(removedIndex, 1);
      if (removedIndex === 0) {
        invalidatePreviewSource();
      }
      renderFiles();
      scheduleAutoPreview();
    });
  });
}

function renderTrainingFiles() {
  if (!state.trainingFiles.length) {
    trainingFileList.className = "file-list empty-list";
    trainingFileList.innerHTML = "<p>No training clips selected yet.</p>";
    return;
  }

  const summary = summarizeTrainingFiles(state.trainingFiles, trainingOutputMode?.value || "classic-rvc-support");
  const baseSegment = summary.baseCount ? `${summary.baseName} ${summary.baseCount} | ` : "";
  const pairSummary = summary.requiresPairs
    ? `${summary.targetName} ${summary.targetCount} | ${summary.sourceName} ${summary.sourceCount} | matched ${summary.targetName}/${summary.sourceName} pairs ${summary.matchedPairCount}`
    : `${summary.targetName} ${summary.targetCount} | ${summary.sourceName} ${summary.sourceCount} | no exact pairs required`;
  const mismatchSummary = summary.requiresPairs
    ? (
      summary.unmatchedTargets.length || summary.unmatchedSources.length
        ? `Unmatched names: ${summary.unmatchedTargets.length} ${summary.targetName} and ${summary.unmatchedSources.length} ${summary.sourceName} still need partners.`
        : `All discovered ${summary.targetName} and ${summary.sourceName} names currently have partners.`
    )
    : (
      summary.sourceCount
        ? `${summary.sourceName} clips will be kept as fixed checkpoint audition sources instead of target-speaker truth audio.`
        : `Add ${summary.sourceName} clips if you want checkpoint auditions to use that source domain instead of falling back to target truth audio.`
    );
  trainingFileList.className = "file-list";
  trainingFileList.innerHTML = `
    <div class="result-row">
      <div class="result-name">${summary.label} dataset summary</div>
      <div class="result-meta">${baseSegment}${pairSummary}</div>
      <div class="result-meta">${mismatchSummary}</div>
    </div>
  `;
  state.trainingFiles.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `
      <div>
        <div class="file-name">${file.name}</div>
        <div class="file-meta">${formatBytes(file.size)}</div>
      </div>
      <button class="remove-file" type="button" data-index="${index}">Remove</button>
    `;
    trainingFileList.appendChild(row);
  });

  trainingFileList.querySelectorAll(".remove-file").forEach((button) => {
    button.addEventListener("click", () => {
      state.trainingFiles.splice(Number(button.dataset.index), 1);
      renderTrainingFiles();
    });
  });
}

function renderTrainingPlanFiles() {
  if (!state.trainingPlanFiles.length) {
    trainingPlanFileList.className = "file-list empty-list";
    trainingPlanFileList.innerHTML = "<p>No persona plan uploaded yet.</p>";
    return;
  }

  trainingPlanFileList.className = "file-list";
  trainingPlanFileList.innerHTML = "";
  state.trainingPlanFiles.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `
      <div>
        <div class="file-name">${file.name}</div>
        <div class="file-meta">${formatBytes(file.size)}</div>
      </div>
      <button class="remove-file" type="button" data-index="${index}">Remove</button>
    `;
    trainingPlanFileList.appendChild(row);
  });

  trainingPlanFileList.querySelectorAll(".remove-file").forEach((button) => {
    button.addEventListener("click", () => {
      state.trainingPlanFiles.splice(Number(button.dataset.index), 1);
      renderTrainingPlanFiles();
    });
  });
}

function renderTrainingTranscriptFiles() {
  if (!state.trainingTranscriptFiles.length) {
    trainingTranscriptFileList.className = "file-list empty-list";
    trainingTranscriptFileList.innerHTML = "<p>No transcript files selected yet.</p>";
    return;
  }

  trainingTranscriptFileList.className = "file-list";
  trainingTranscriptFileList.innerHTML = "";
  state.trainingTranscriptFiles.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `
      <div>
        <div class="file-name">${file.name}</div>
        <div class="file-meta">${formatBytes(file.size)}</div>
      </div>
      <button class="remove-file" type="button" data-index="${index}">Remove</button>
    `;
    trainingTranscriptFileList.appendChild(row);
  });

  trainingTranscriptFileList.querySelectorAll(".remove-file").forEach((button) => {
    button.addEventListener("click", () => {
      state.trainingTranscriptFiles.splice(Number(button.dataset.index), 1);
      renderTrainingTranscriptFiles();
    });
  });
}

function renderIndexFile() {
  if (!state.indexFile) {
    indexFileList.className = "file-list empty-list";
    indexFileList.innerHTML = "<p>No index file selected. Auto-detect stays on.</p>";
    return;
  }

  indexFileList.className = "file-list";
  indexFileList.innerHTML = `
    <div class="file-row">
      <div>
        <div class="file-name">${state.indexFile.name}</div>
        <div class="file-meta">${formatBytes(state.indexFile.size)}</div>
      </div>
      <button id="removeIndexFile" class="remove-file" type="button">Remove</button>
    </div>
  `;
  document.getElementById("removeIndexFile").addEventListener("click", () => {
    state.indexFile = null;
    renderIndexFile();
    scheduleAutoPreview();
  });
}

function renderGenerateGuideFile() {
  renderSingleFileCard(
    generateFileList,
    state.generateGuideFile,
    "No reference vocal selected yet.",
    "removeGenerateGuideFile",
    () => {
      state.generateGuideFile = null;
      renderGenerateGuideFile();
    },
  );
}

function renderDetagFile() {
  if (!state.detagFile) {
    detagFileList.className = "file-list empty-list";
    detagFileList.innerHTML = "<p>No file selected yet.</p>";
    return;
  }

  detagFileList.className = "file-list";
  detagFileList.innerHTML = `
    <div class="file-row">
      <div>
        <div class="file-name">${state.detagFile.name}</div>
        <div class="file-meta">${formatBytes(state.detagFile.size)}</div>
      </div>
      <button id="removeDetagFile" class="remove-file" type="button">Remove</button>
    </div>
  `;
  document.getElementById("removeDetagFile").addEventListener("click", () => {
    state.detagFile = null;
    renderDetagFile();
  });
}

function renderIsolatorFile() {
  if (!state.isolatorFiles.length) {
    isolatorFileList.className = "file-list empty-list";
    isolatorFileList.innerHTML = "<p>No files selected yet.</p>";
    return;
  }

  isolatorFileList.className = "file-list";
  isolatorFileList.innerHTML = "";
  state.isolatorFiles.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `
      <div>
        <div class="file-name">${file.name}</div>
        <div class="file-meta">${formatBytes(file.size)}</div>
      </div>
      <button class="remove-file" type="button" data-index="${index}">Remove</button>
    `;
    isolatorFileList.appendChild(row);
  });

  isolatorFileList.querySelectorAll(".remove-file").forEach((button) => {
    button.addEventListener("click", () => {
      state.isolatorFiles.splice(Number(button.dataset.index), 1);
      renderIsolatorFile();
    });
  });
}

function renderMasteringSourceFile() {
  if (!state.masteringSourceFile) {
    masteringSourceFileList.className = "file-list empty-list";
    masteringSourceFileList.innerHTML = "<p>No source file selected yet.</p>";
    return;
  }

  masteringSourceFileList.className = "file-list";
  masteringSourceFileList.innerHTML = `
    <div class="file-row">
      <div>
        <div class="file-name">${state.masteringSourceFile.name}</div>
        <div class="file-meta">${formatBytes(state.masteringSourceFile.size)}</div>
      </div>
      <button id="removeMasteringSourceFile" class="remove-file" type="button">Remove</button>
    </div>
  `;
  document.getElementById("removeMasteringSourceFile").addEventListener("click", () => {
    state.masteringSourceFile = null;
    renderMasteringSourceFile();
  });
}

function renderMasteringReferenceFile() {
  if (!state.masteringReferenceFiles.length) {
    masteringReferenceFileList.className = "file-list empty-list";
    masteringReferenceFileList.innerHTML = "<p>No reference files selected yet.</p>";
    return;
  }

  masteringReferenceFileList.className = "file-list";
  masteringReferenceFileList.innerHTML = "";
  state.masteringReferenceFiles.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `
      <div>
        <div class="file-name">${file.name}</div>
        <div class="file-meta">${formatBytes(file.size)}</div>
      </div>
      <button class="remove-file" type="button" data-index="${index}">Remove</button>
    `;
    masteringReferenceFileList.appendChild(row);
  });

  masteringReferenceFileList.querySelectorAll(".remove-file").forEach((button) => {
    button.addEventListener("click", () => {
      state.masteringReferenceFiles.splice(Number(button.dataset.index), 1);
      renderMasteringReferenceFile();
    });
  });
}

function renderOptimizeFiles() {
  if (!state.optimizeFiles.length) {
    optimizeFileList.className = "file-list empty-list";
    optimizeFileList.innerHTML = "<p>No files selected yet.</p>";
    return;
  }

  optimizeFileList.className = "file-list";
  optimizeFileList.innerHTML = "";
  state.optimizeFiles.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `
      <div>
        <div class="file-name">${file.name}</div>
        <div class="file-meta">${formatBytes(file.size)}</div>
      </div>
      <button class="remove-file" type="button" data-index="${index}">Remove</button>
    `;
    optimizeFileList.appendChild(row);
  });

  optimizeFileList.querySelectorAll(".remove-file").forEach((button) => {
    button.addEventListener("click", () => {
      state.optimizeFiles.splice(Number(button.dataset.index), 1);
      renderOptimizeFiles();
    });
  });
}

function renderAlbumBulkFiles() {
  if (!state.albumBulkFiles.length) {
    albumBulkFileList.className = "file-list empty-list";
    albumBulkFileList.innerHTML = "<p>No files selected yet.</p>";
    return;
  }

  albumBulkFileList.className = "file-list";
  albumBulkFileList.innerHTML = "";
  state.albumBulkFiles.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `
      <div>
        <div class="file-name">${file.name}</div>
        <div class="file-meta">${formatBytes(file.size)}</div>
      </div>
      <button class="remove-file" type="button" data-index="${index}">Remove</button>
    `;
    albumBulkFileList.appendChild(row);
  });

  albumBulkFileList.querySelectorAll(".remove-file").forEach((button) => {
    button.addEventListener("click", () => {
      state.albumBulkFiles.splice(Number(button.dataset.index), 1);
      renderAlbumBulkFiles();
    });
  });
}

function setAlbumBulkFiles(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  for (const file of incoming) {
    const alreadyAdded = state.albumBulkFiles.some(
      (existing) => existing.name === file.name && existing.size === file.size,
    );
    if (!alreadyAdded) {
      state.albumBulkFiles.push(file);
    }
  }
  renderAlbumBulkFiles();
}

function renderAlbumProjectSelector() {
  const projects = Array.isArray(state.albumProjects) ? state.albumProjects : [];
  if (!projects.length) {
    albumProjectSelect.innerHTML = `<option value="">No album projects yet</option>`;
    return;
  }

  albumProjectSelect.innerHTML = projects
    .map((project) => {
      const filled = Number(project.filled_songs || 0);
      const label = `${project.name} (${filled} tracks)`;
      return `<option value="${project.id}">${label}</option>`;
    })
    .join("");

  if (state.currentAlbumProjectId) {
    albumProjectSelect.value = state.currentAlbumProjectId;
  } else {
    albumProjectSelect.value = projects[0].id;
    state.currentAlbumProjectId = projects[0].id;
  }
}

function renderAlbumProject(project) {
  state.albumCurrentProject = project || null;
  if (!project) {
    albumResultSummary.classList.add("hidden");
    albumResultSummary.innerHTML = "";
    albumSongsList.className = "album-track-body empty-list";
    albumSongsList.innerHTML = "<p>Create or load an album project to map tracks.</p>";
    albumLogList.className = "results-list empty-list";
    albumLogList.innerHTML = "<p>No log entries yet.</p>";
    albumMixCard.classList.add("hidden");
    albumPreviewPlayer.removeAttribute("src");
    albumPreviewDownload.setAttribute("href", "#");
    albumNowPlayingTitle.textContent = "Nothing playing";
    albumNowPlayingMeta.textContent = "Click a track title or a version from its dropdown.";
    return;
  }

  const filledSongs = Number(project.filled_songs || 0);
  albumResultSummary.classList.remove("hidden");
  albumResultSummary.innerHTML = `
    <div class="album-summary-kicker">Album</div>
    <strong>${project.name}</strong>
    <p>${filledSongs} track${filledSongs === 1 ? "" : "s"} ready. Click any title to play the latest version and drag rows to reorder the album.</p>
  `;

  if (project.latest_mix_url) {
    albumMixCard.classList.remove("hidden");
    albumPreviewDownload.href = project.latest_mix_url;
    albumPreviewDownload.download = project.latest_mix_download_name || "album_preview.wav";
    albumPlayMixButton.disabled = false;
  } else {
    albumMixCard.classList.add("hidden");
    albumPreviewDownload.href = "#";
    albumPlayMixButton.disabled = true;
  }

  const songs = Array.isArray(project.songs) ? project.songs : [];
  if (!songs.length) {
    albumSongsList.className = "album-track-body empty-list";
    albumSongsList.innerHTML = "<p>No tracks yet. Upload tracks above to build this album.</p>";
  } else {
    albumSongsList.className = "album-track-body";
    albumSongsList.innerHTML = songs
      .map((song) => {
        const songIndex = Number(song.song_index || 0);
        const versions = Array.isArray(song.versions) ? song.versions : [];
        const latest = song.latest_version || versions[versions.length - 1] || null;
        const titleText = song.title || `Song ${songIndex}`;
        const dateText = latest ? formatRelativeTime(latest.created_at) : "--";
        const durationText = latest && latest.url ? formatDuration(latest.duration_seconds) : "--";
        const dropdownSummary = latest ? `V${Number(latest.version || 0)}` : "Upload";
        const versionMenu = versions.length
          ? versions
              .slice()
              .reverse()
              .map((version) => {
                const versionNumber = Number(version.version || 0);
                const versionMeta = `${formatRelativeTime(version.created_at)} | ${formatDuration(version.duration_seconds)}`;
                const playable = Boolean(version.url);
                return `
                  <button
                    class="album-version-play"
                    data-url="${version.url}"
                    data-title="${titleText}"
                    data-meta="Version V${versionNumber} | ${versionMeta}"
                    ${playable ? "" : "disabled"}
                    type="button"
                  >
                    <span>V${versionNumber}</span>
                    <span>${playable ? versionMeta : "Missing file"}</span>
                  </button>
                `;
              })
              .join("")
          : `<div class="album-version-empty">No past versions yet.</div>`;

        return `
          <div class="album-track-row" data-song-index="${songIndex}">
            <div class="album-col-index">
              <button class="album-drag-handle" type="button" draggable="true" aria-label="Drag track ${songIndex}">⋮⋮</button>
              <span>${songIndex}</span>
            </div>
            <div class="album-col-title">
              ${
                latest && latest.url
                  ? `<button class="album-track-play" data-url="${latest.url}" data-title="${titleText}" data-meta="Latest version | V${Number(latest.version || 0)} | ${durationText}">${titleText}</button>`
                  : `<span class="album-track-title-placeholder">${titleText}</span>`
              }
              <details class="album-version-dropdown">
                <summary>${dropdownSummary}</summary>
                <div class="album-version-menu">
                  ${versionMenu}
                  <label class="album-version-replace" for="albumSongFile-${songIndex}">${latest ? "Replace track" : "Upload first version"}</label>
                  <input id="albumSongFile-${songIndex}" class="album-version-file-input" data-song-index="${songIndex}" type="file" accept="audio/*" hidden>
                </div>
              </details>
            </div>
            <div class="album-col-date">${dateText}</div>
            <div class="album-col-duration">
              <span>${durationText}</span>
              <button class="album-track-delete" data-song-index="${songIndex}" type="button" aria-label="Delete track ${songIndex}">Delete</button>
            </div>
          </div>
        `;
      })
      .join("");

    albumSongsList.querySelectorAll(".album-track-play").forEach((button) => {
      button.addEventListener("click", () => {
        playAlbumSource(
          button.dataset.url || "",
          button.dataset.title || "Track",
          button.dataset.meta || "",
          true,
        );
      });
    });

    albumSongsList.querySelectorAll(".album-version-play").forEach((button) => {
      button.addEventListener("click", async () => {
        playAlbumSource(
          button.dataset.url || "",
          button.dataset.title || "Track version",
          button.dataset.meta || "",
          true,
        );
        const dropdown = button.closest("details");
        if (dropdown) {
          dropdown.open = false;
        }
      });
    });

    albumSongsList.querySelectorAll(".album-version-file-input").forEach((input) => {
      input.addEventListener("change", async () => {
        const file = input.files?.[0];
        const songIndex = Number(input.dataset.songIndex || 0);
        if (!file || !songIndex) {
          return;
        }
        await uploadAlbumSongVersion(songIndex, file);
        input.value = "";
        const dropdown = input.closest("details");
        if (dropdown) {
          dropdown.open = false;
        }
      });
    });

    albumSongsList.querySelectorAll(".album-track-delete").forEach((button) => {
      button.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        const songIndex = Number(button.dataset.songIndex || 0);
        if (!songIndex) {
          return;
        }
        const songName = button.closest(".album-track-row")?.querySelector(".album-track-play, .album-track-title-placeholder")?.textContent || `Track ${songIndex}`;
        setAlbumStatus("running", "Deleting track", `Removing ${songName} and rebuilding the album preview...`, 45);
        try {
          const response = await fetch(`/api/albums/projects/${state.currentAlbumProjectId}/songs/${songIndex}`, {
            method: "DELETE",
          });
          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload.detail || "Could not delete track.");
          }
          renderAlbumProject(payload.project || null);
          await loadAlbumProjects(false);
          renderAlbumProjectSelector();
          setAlbumStatus("completed", "Track deleted", `${songName} was removed from the album.`, 100);
        } catch (error) {
          setAlbumStatus("failed", "Could not delete track", error.message || String(error), 0);
        }
      });
    });

    wireAlbumTrackReorder();
  }

  if (albumPlayMixButton) {
    albumPlayMixButton.onclick = () => {
      if (!project.latest_mix_url) {
        return;
      }
      playAlbumSource(
        project.latest_mix_url,
        `${project.name} | Album mix`,
        `${filledSongs} current track${filledSongs === 1 ? "" : "s"} | 0.5s crossfade`,
        true,
      );
    };
  }

  const logEntries = Array.isArray(project.event_log) ? project.event_log : [];
  if (!logEntries.length) {
    albumLogList.className = "results-list empty-list";
    albumLogList.innerHTML = "<p>No log entries yet.</p>";
  } else {
    albumLogList.className = "results-list";
    albumLogList.innerHTML = logEntries
      .slice(-120)
      .reverse()
      .map(
        (entry) => `
          <div class="result-row">
            <div class="result-meta">${entry.at || ""}</div>
            <div>${entry.message || ""}</div>
          </div>
        `,
      )
      .join("");
  }
}

function renderSingleFileCard(targetElement, file, emptyMessage, removeId, onRemove) {
  if (!file) {
    targetElement.className = "file-list empty-list";
    targetElement.innerHTML = `<p>${emptyMessage}</p>`;
    return;
  }

  targetElement.className = "file-list";
  targetElement.innerHTML = `
    <div class="file-row">
      <div>
        <div class="file-name">${file.name}</div>
        <div class="file-meta">${formatBytes(file.size)}</div>
      </div>
      <button id="${removeId}" class="remove-file" type="button">Remove</button>
    </div>
  `;
  document.getElementById(removeId).addEventListener("click", onRemove);
}

function renderApiComposeMidiFile() {
  renderSingleFileCard(
    apiComposeMidiFileList,
    state.apiComposeMidiFile,
    "No MIDI file selected yet.",
    "removeApiComposeMidiFile",
    () => {
      state.apiComposeMidiFile = null;
      renderApiComposeMidiFile();
    },
  );
}

function renderApiComposeBeatFile() {
  renderSingleFileCard(
    apiComposeBeatFileList,
    state.apiComposeBeatFile,
    "No beat file selected.",
    "removeApiComposeBeatFile",
    () => {
      state.apiComposeBeatFile = null;
      renderApiComposeBeatFile();
    },
  );
}

function renderTouchUpFile(targetElement, file, emptyMessage, removeId, onRemove) {
  if (!targetElement) {
    return;
  }
  if (!file) {
    targetElement.className = "file-list empty-list";
    targetElement.innerHTML = `<p>${emptyMessage}</p>`;
    return;
  }

  targetElement.className = "file-list";
  targetElement.innerHTML = `
    <div class="file-row">
      <div>
        <div class="file-name">${file.name}</div>
        <div class="file-meta">${formatBytes(file.size)}</div>
      </div>
      <button id="${removeId}" class="remove-file" type="button">Remove</button>
    </div>
  `;
  document.getElementById(removeId).addEventListener("click", onRemove);
}

function renderTouchUpReferenceFile() {
  renderTouchUpFile(
    touchUpReferenceFileList,
    state.touchUpReferenceFile,
    "No reference file needed for the current method.",
    "removeTouchUpReferenceFile",
    () => {
      state.touchUpReferenceFile = null;
      renderTouchUpReferenceFile();
    },
  );
}

function renderTouchUpSourceFile() {
  renderTouchUpFile(
    touchUpSourceFileList,
    state.touchUpSourceFile,
    "No AI vocal file selected yet.",
    "removeTouchUpSourceFile",
    () => {
      state.touchUpSourceFile = null;
      renderTouchUpSourceFile();
    },
  );
}

function addFiles(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  const picked = incoming[0] || null;
  const changed =
    !picked
      ? false
      : !state.files.length
        || state.files[0].name !== picked.name
        || state.files[0].size !== picked.size;
  state.files = picked ? [picked] : [];
  renderFiles();
  if (changed) {
    invalidatePreviewSource();
    scheduleAutoPreview();
  }
}

function addTrainingFiles(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  for (const file of incoming) {
    const alreadyAdded = state.trainingFiles.some(
      (existing) => existing.name === file.name && existing.size === file.size,
    );
    if (!alreadyAdded) {
      state.trainingFiles.push(file);
    }
  }
  renderTrainingFiles();
}

function addTrainingPlanFiles(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  for (const file of incoming) {
    const alreadyAdded = state.trainingPlanFiles.some(
      (existing) => existing.name === file.name && existing.size === file.size,
    );
    if (!alreadyAdded) {
      state.trainingPlanFiles.push(file);
    }
  }
  renderTrainingPlanFiles();
}

function addTrainingTranscriptFiles(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  for (const file of incoming) {
    const alreadyAdded = state.trainingTranscriptFiles.some(
      (existing) => existing.name === file.name && existing.size === file.size,
    );
    if (!alreadyAdded) {
      state.trainingTranscriptFiles.push(file);
    }
  }
  renderTrainingTranscriptFiles();
}

function setDetagFile(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  state.detagFile = incoming[0] || null;
  renderDetagFile();
}

function setIsolatorFile(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  for (const file of incoming) {
    const alreadyAdded = state.isolatorFiles.some(
      (existing) => existing.name === file.name && existing.size === file.size,
    );
    if (!alreadyAdded) {
      state.isolatorFiles.push(file);
    }
  }
  renderIsolatorFile();
}

function setMasteringSourceFile(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  state.masteringSourceFile = incoming[0] || null;
  renderMasteringSourceFile();
}

function setGenerateGuideFile(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  state.generateGuideFile = incoming[0] || null;
  renderGenerateGuideFile();
}

function setMasteringReferenceFile(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  for (const file of incoming) {
    const alreadyAdded = state.masteringReferenceFiles.some(
      (existing) => existing.name === file.name && existing.size === file.size,
    );
    if (!alreadyAdded) {
      state.masteringReferenceFiles.push(file);
    }
  }
  renderMasteringReferenceFile();
}

function setOptimizeFiles(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  for (const file of incoming) {
    const alreadyAdded = state.optimizeFiles.some(
      (existing) => existing.name === file.name && existing.size === file.size,
    );
    if (!alreadyAdded) {
      state.optimizeFiles.push(file);
    }
  }
  renderOptimizeFiles();
}

function setApiComposeMidiFile(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  state.apiComposeMidiFile = incoming[0] || null;
  renderApiComposeMidiFile();
}

function setApiComposeBeatFile(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  state.apiComposeBeatFile = incoming[0] || null;
  renderApiComposeBeatFile();
}

function setTouchUpReferenceFile(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  state.touchUpReferenceFile = incoming[0] || null;
  renderTouchUpReferenceFile();
}

function setTouchUpSourceFile(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  state.touchUpSourceFile = incoming[0] || null;
  renderTouchUpSourceFile();
}

function setIndexFile(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  const picked = incoming[0] || null;
  if (picked && !picked.name.toLowerCase().endsWith(".index")) {
    setStatus(
      "failed",
      "Wrong file type",
      "Please drop a real .index file here.",
      0,
    );
    return;
  }
  state.indexFile = picked;
  renderIndexFile();
  scheduleAutoPreview();
}

function renderModels() {
  const populate = (selectEl, modelsToRender, { showEmptyState = false, emptyMessage = "No persona voices found" } = {}) => {
    if (!selectEl) {
      return;
    }
    const currentValue = selectEl.value;
    selectEl.innerHTML = "";
    if (!modelsToRender.length) {
      selectEl.innerHTML = `<option value="">${emptyMessage}</option>`;
      if (showEmptyState && modelEmptyState) {
        modelEmptyState.classList.remove("hidden");
        modelEmptyState.innerHTML =
          "<strong>No conversion voices found yet.</strong><p>Add a classic RVC <code>.pth</code> voice to <code>weights/</code> or train a paired aligned package from BASE, TARGET, and SUNO clips, then refresh this list.</p>";
      }
      return;
    }

    if (showEmptyState && modelEmptyState) {
      modelEmptyState.classList.add("hidden");
    }
    for (const model of modelsToRender) {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.label || model.name;
      option.dataset.indexPath = model.default_index || "";
      selectEl.appendChild(option);
    }
    if (modelsToRender.some((model) => model.name === currentValue)) {
      selectEl.value = currentValue;
    } else {
      selectEl.value = modelsToRender[0].name;
    }
  };

  const leadBuilderModels = getLeadBuilderModels();
  populate(modelSelect, leadBuilderModels, {
    showEmptyState: true,
    emptyMessage: "No conversion voices found",
  });
  populate(secondaryModelSelect, leadBuilderModels);
  populate(generateModelSelect, getPersonaRepairModels());
  if (pipaPackageSelect) {
    const selected = getLeadBuilderSelectedModel();
    pipaPackageSelect.innerHTML = `<option value="${selected?.name || ""}">${selected?.label || selected?.system || "Selected conversion voice"}</option>`;
    if (selected?.name) {
      pipaPackageSelect.value = selected.name;
    }
  }
  if (
    secondaryModelSelect &&
    leadBuilderModels.length > 1 &&
    secondaryModelSelect.value === modelSelect.value
  ) {
    const alternate = leadBuilderModels.find((model) => model.name !== modelSelect.value);
    if (alternate) {
      secondaryModelSelect.value = alternate.name;
    }
  }
  syncModelDefaults();
  updatePipaPackageSummary();
}

function renderDetagVoices() {
  detagVoiceSelect.innerHTML = "";
  if (!state.detagVoices.length) {
    detagEmptyState.classList.remove("hidden");
    detagEmptyState.innerHTML =
      "<strong>No detag voices found yet.</strong><p>Add a model to weights or keep a reference folder in logs/&lt;voice&gt;/0_gt_wavs.</p>";
    detagVoiceSelect.innerHTML = `<option value="">No voices found</option>`;
    return;
  }

  detagEmptyState.classList.add("hidden");
  detagEmptyState.innerHTML = "";
  let firstReadyValue = "";
  for (const voice of state.detagVoices) {
    const option = document.createElement("option");
    option.value = voice.id;
    const clipLabel = voice.reference_clips
      ? `${voice.reference_clips} ref clips`
      : "no ref clips yet";
    option.textContent = `${voice.label || voice.id} (${clipLabel})`;
    option.disabled = !voice.ready;
    if (voice.ready && !firstReadyValue) {
      firstReadyValue = voice.id;
    }
    detagVoiceSelect.appendChild(option);
  }

  if (firstReadyValue) {
    detagVoiceSelect.value = firstReadyValue;
  }

  const hasUnreadyVoices = state.detagVoices.some((voice) => !voice.ready);
  if (hasUnreadyVoices) {
    detagEmptyState.classList.remove("hidden");
    detagEmptyState.innerHTML =
      "<strong>Some voices need references first.</strong><p>Models from the weights folder now appear here, but detag only works when matching clips still exist in logs/&lt;voice&gt;/0_gt_wavs.</p>";
  }
}

function syncModelDefaults() {
  const selected = getLeadBuilderSelectedModel();
  if (!selected) {
    indexPath.value = "";
    return;
  }
  modelSelect.value = selected.name;
  indexPath.value = selected.default_index || "";
  updatePipaPackageSummary();
}

function updatePipaPackageSummary() {
  if (!pipaPackageSummary) {
    return;
  }
  const selectedModel = getLeadBuilderSelectedModel();
  if (!selectedModel) {
    pipaPackageSummary.textContent =
      "Paired aligned and classic RVC .pth voices appear here. Add one to use Lead Builder.";
    return;
  }
  pipaPackageSummary.textContent =
    `${selectedModel.label || selectedModel.name} will run as a direct ${String(selectedModel.system || "conversion").toLowerCase()} run. Lyrics, repair modes, and rebuild passes are not used here.`;
}

function updateQualitySummary() {
  const preset = state.qualityPresets[qualityPreset.value];
  if (!preset) {
    qualitySummary.textContent =
      "Balanced is the safest default for most direct paired conversions.";
    return;
  }
  qualitySummary.textContent = `${preset.label}: ${preset.description}`;
}

function updateMasterProfileSummary() {
  if (!masterProfileSummary || !masterProfile) {
    return;
  }
  const profiles = Array.isArray(state.masterOptions?.profiles)
    ? state.masterOptions.profiles
    : [];
  const selected = profiles.find((profile) => profile.id === masterProfile.value);
  if (!selected) {
    masterProfileSummary.textContent =
      "Balances target lead tone, weak-phrase cleanup, and non-lyric removal.";
    return;
  }
  masterProfileSummary.textContent = selected.description || selected.label || "";
}

function updateGenerateQualitySummary() {
  if (!generateQualitySummary || !generateQualityPreset) {
    return;
  }
  const preset = state.qualityPresets[generateQualityPreset.value];
  if (!preset) {
    generateQualitySummary.textContent =
      "Balanced runs a few pronunciation repair passes after conversion and is the best starting point.";
    return;
  }
  const repairSummary = {
    fast: "Fewer repair passes for quick testing.",
    balanced: "A good balance between speed and lyric repair depth.",
    clean: "More repair passes and wider word targeting for the best shot at cleaner diction.",
  };
  generateQualitySummary.textContent = `${preset.label}: ${repairSummary[generateQualityPreset.value] || preset.description}`;
}

function applyQualityPresetDefaults() {
  const preset = state.qualityPresets[qualityPreset.value];
  if (!preset) {
    return;
  }

  pitchMethod.value = preset.f0_method || "";
  indexRate.value = String(Math.round(Number(preset.index_rate || 0) * 100));
  protect.value = String(Math.round(Number(preset.protect || 0) * 100));
  rmsMixRate.value = String(Math.round(Number(preset.rms_mix_rate || 0) * 100));
  filterRadius.value = String(Number(preset.filter_radius || 0));
  updateSliderLabels();
}

function updateIsolatorModeSummary() {
  const modes = state.isolatorOptions?.modes || [];
  const selected = modes.find((mode) => mode.id === isolatorMode.value);
  if (!selected) {
    isolatorModeSummary.textContent =
      "Best for splitting a vocal stem into lead and backing layers.";
    return;
  }
  isolatorModeSummary.textContent = selected.description;
}

function updateIsolatorInputTypeSummary() {
  const types = state.isolatorOptions?.input_types || [];
  const selected = types.find((inputType) => inputType.id === isolatorInputType.value);
  if (!selected) {
    isolatorInputTypeSummary.textContent =
      "Use this when vocals are still sitting on top of music or a beat.";
    return;
  }
  isolatorInputTypeSummary.textContent = selected.description;
}

function renderTrainingPackageDownloads() {
  if (!trainingPackageDownloadSelect) {
    return;
  }

  const packages = Array.isArray(state.pipaPackages)
    ? state.pipaPackages.filter((entry) => isDownloadableTrainingPackageMode(entry?.package_mode || "persona-v1"))
    : [];
  const previousValue = trainingPackageDownloadSelect.value;
  trainingPackageDownloadSelect.innerHTML = "";

  if (!packages.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No training packages found yet";
    trainingPackageDownloadSelect.appendChild(option);
    trainingPackageDownloadSelect.disabled = true;
    if (downloadTrainingPackageButton) {
      downloadTrainingPackageButton.disabled = true;
    }
    if (trainingPackageDownloadSummary) {
      trainingPackageDownloadSummary.textContent = "Train a voice package first, then download the full folder here.";
    }
    return;
  }

  packages.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.name || "";
    option.textContent = entry.label || entry.name || "Training package";
    trainingPackageDownloadSelect.appendChild(option);
  });

  const selectedValue = packages.some((entry) => entry.name === previousValue)
    ? previousValue
    : (packages[0]?.name || "");
  trainingPackageDownloadSelect.value = selectedValue;
  trainingPackageDownloadSelect.disabled = false;
  updateTrainingPackageDownloadSummary();
}

function updateTrainingPackageDownloadSummary() {
  const selectionName = trainingPackageDownloadSelect?.value || "";
  if (downloadTrainingPackageButton) {
    downloadTrainingPackageButton.disabled = !selectionName;
  }
  if (!trainingPackageDownloadSummary) {
    return;
  }
  const selected = (state.pipaPackages || []).find((entry) => entry.name === selectionName);
  trainingPackageDownloadSummary.textContent = selected
    ? `Downloads ${selected.label || selected.name} as a full training package zip.`
    : "Pick a training package to download its full folder.";
}

function updateTrainingResumeSummary() {
  if (!trainingResumeSummary) {
    return;
  }
  const selectionName = trainingResumePackageSelect?.value || "";
  const selected = (state.pipaPackages || []).find((entry) => entry.name === selectionName);
  if (!selected) {
    trainingResumeSummary.textContent =
      "Start fresh, or pick a Persona package to continue from its latest saved checkpoint.";
    return;
  }
  const bestEpoch = Number(selected.guided_regeneration_best_epoch || 0);
  const lastEpoch = Number(selected.guided_regeneration_last_epoch || 0);
  const bestTotal = Number(selected.guided_regeneration_best_val_total || 0);
  const quality = selected.guided_regeneration_quality_summary
    ? ` | ${selected.guided_regeneration_quality_summary}`
    : "";
  trainingResumeSummary.textContent =
    `${selected.label || selected.name} will resume from its latest saved checkpoint. ` +
    `Best epoch ${bestEpoch || "unknown"} | last epoch ${lastEpoch || "unknown"} | ` +
    `best total ${bestTotal ? bestTotal.toFixed(4) : "unknown"}${quality}`;
}

function renderTrainingResumeOptions() {
  if (!trainingResumePackageSelect) {
    return;
  }

  const packages = Array.isArray(state.pipaPackages)
    ? state.pipaPackages.filter((entry) => isResumablePersonaPackageMode(entry?.package_mode || "persona-v1"))
    : [];
  const previousValue = trainingResumePackageSelect.value;
  trainingResumePackageSelect.innerHTML = "";

  const freshOption = document.createElement("option");
  freshOption.value = "";
  freshOption.textContent = "Start fresh (no saved package)";
  trainingResumePackageSelect.appendChild(freshOption);

  packages.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.name || "";
    const bestEpoch = Number(entry.guided_regeneration_best_epoch || 0);
    const lastEpoch = Number(entry.guided_regeneration_last_epoch || 0);
    option.textContent = `${entry.label || entry.name || "Persona package"}${bestEpoch || lastEpoch ? ` (best ${bestEpoch || "?"}, last ${lastEpoch || "?"})` : ""}`;
    trainingResumePackageSelect.appendChild(option);
  });

  trainingResumePackageSelect.value = packages.some((entry) => entry.name === previousValue)
    ? previousValue
    : "";
  updateTrainingResumeSummary();
  updateTrainingCurriculumSummary();
}

function downloadSelectedTrainingPackage() {
  if (!trainingPackageDownloadSelect || !trainingPackageDownloadSelect.value) {
    if (trainingPackageDownloadSummary) {
      trainingPackageDownloadSummary.textContent = "Pick a training package before downloading.";
    }
    return;
  }

  const selectionName = trainingPackageDownloadSelect.value;
  const selected = (state.pipaPackages || []).find((entry) => entry.name === selectionName);
  if (trainingPackageDownloadSummary) {
    trainingPackageDownloadSummary.textContent = selected
      ? `Preparing ${selected.label || selectionName} for download...`
      : "Preparing training package download...";
  }

  const link = document.createElement("a");
  link.href = `/api/training/packages/download?selection_name=${encodeURIComponent(selectionName)}`;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

async function loadModels() {
  const response = await fetch("/api/models");
  const payload = await response.json();
  state.models = payload.models || [];
  state.pipaPackages = Array.isArray(payload.pipa_packages) ? payload.pipa_packages : [];
  state.qualityPresets = payload.quality_presets || {};
  state.preprocessPipelines = Array.isArray(payload.preprocess_pipelines)
    ? payload.preprocess_pipelines
    : [];

  if (preprocessMode && state.preprocessPipelines.length) {
    preprocessMode.innerHTML = "";
    state.preprocessPipelines.forEach((pipeline) => {
      const option = document.createElement("option");
      option.value = pipeline.id;
      option.textContent = pipeline.label;
      preprocessMode.appendChild(option);
    });
    preprocessMode.value =
      state.preprocessPipelines.some((pipeline) => pipeline.id === "off")
        ? "off"
        : state.preprocessPipelines[0].id;
    preprocessMode.disabled = true;
  }

  if (generatePreprocessMode && state.preprocessPipelines.length) {
    const currentValue = generatePreprocessMode.value;
    const defaultMode = payload.preprocess_defaults?.mode || "off";
    generatePreprocessMode.innerHTML = "";
    state.preprocessPipelines.forEach((pipeline) => {
      const option = document.createElement("option");
      option.value = pipeline.id;
      option.textContent = pipeline.label;
      generatePreprocessMode.appendChild(option);
    });
    generatePreprocessMode.value =
      state.preprocessPipelines.some((pipeline) => pipeline.id === currentValue)
        ? currentValue
        : (state.preprocessPipelines.some((pipeline) => pipeline.id === defaultMode)
            ? defaultMode
            : state.preprocessPipelines[0].id);
  }

  if (preprocessStrength) {
    preprocessStrength.value = "1";
    preprocessStrength.disabled = true;
  }
  const defaultPreprocessStrength = Number(payload.preprocess_defaults?.strength);
  if (
    generatePreprocessStrength &&
    Number.isFinite(defaultPreprocessStrength) &&
    (!generatePreprocessStrength.value || generatePreprocessStrength.value === "10")
  ) {
    generatePreprocessStrength.value = String(defaultPreprocessStrength);
  }

  renderModels();
  applyQualityPresetDefaults();
  updateSliderLabels();
  updatePreprocessModeSummary();
  updateGeneratePreprocessModeSummary();
  updateQualitySummary();
  updateGenerateQualitySummary();
  updatePipaPackageSummary();
  renderTrainingPackageDownloads();
  renderTrainingResumeOptions();
}

async function loadMasterConversionOptions() {
  const response = await fetch("/api/master-conversion/options");
  const payload = await response.json();
  state.masterOptions = payload;
  if (masterProfile && Array.isArray(payload.profiles) && payload.profiles.length) {
    const currentValue = masterProfile.value;
    masterProfile.innerHTML = "";
    payload.profiles.forEach((profile) => {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = profile.label;
      masterProfile.appendChild(option);
    });
    masterProfile.value =
      payload.profiles.some((profile) => profile.id === currentValue)
        ? currentValue
        : (payload.defaults?.profile || payload.profiles[0].id);
  }
  if (
    preprocessStrength &&
    payload?.defaults &&
    typeof payload.defaults.candidate_strength !== "undefined" &&
    (!preprocessStrength.value || preprocessStrength.value === "10")
  ) {
    preprocessStrength.value = String(payload.defaults.candidate_strength);
  }
  if (
    qualityPreset &&
    payload?.defaults &&
    payload.defaults.quality_preset &&
    state.qualityPresets[payload.defaults.quality_preset]
  ) {
    qualityPreset.value = payload.defaults.quality_preset;
  }
  updateSliderLabels();
  updateMasterProfileSummary();
  updateQualitySummary();
}

async function loadGenerateOptions() {
  const response = await fetch("/api/generate/options");
  const payload = await response.json();
  state.generateOptions = payload;

  const populateKeySelect = (selectEl) => {
    if (!selectEl || !Array.isArray(payload.keys) || !payload.keys.length) {
      return;
    }
    const currentValue = selectEl.value;
    selectEl.innerHTML = "";
    payload.keys.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = entry.label;
      selectEl.appendChild(option);
    });
    selectEl.value = payload.keys.some((entry) => entry.id === currentValue) ? currentValue : "";
  };

  populateKeySelect(generateGuideKey);
  populateKeySelect(generateTargetKey);

  if (payload?.defaults) {
    if (generateQualityPreset && payload.defaults.quality_preset) {
      generateQualityPreset.value = payload.defaults.quality_preset;
    }
    if (
      generatePreprocessMode &&
      payload.defaults.preprocess_mode &&
      Array.isArray(state.preprocessPipelines) &&
      state.preprocessPipelines.some((pipeline) => pipeline.id === payload.defaults.preprocess_mode)
    ) {
      generatePreprocessMode.value = payload.defaults.preprocess_mode;
    }
    if (
      generatePreprocessStrength &&
      typeof payload.defaults.preprocess_strength !== "undefined"
    ) {
      generatePreprocessStrength.value = String(payload.defaults.preprocess_strength);
    }
  }

  updateSliderLabels();
  updateGeneratePreprocessModeSummary();
  updateGenerateQualitySummary();
}

async function loadIsolatorOptions() {
  const response = await fetch("/api/isolator/options");
  const payload = await response.json();
  state.isolatorOptions = payload;

  if (Array.isArray(payload.modes) && payload.modes.length) {
    const currentValue = isolatorMode.value;
    isolatorMode.innerHTML = "";
    payload.modes.forEach((mode) => {
      const option = document.createElement("option");
      option.value = mode.id;
      option.textContent = mode.label;
      isolatorMode.appendChild(option);
    });
    isolatorMode.value = payload.defaults?.mode || currentValue || payload.modes[0].id;
  }

  if (Array.isArray(payload.input_types) && payload.input_types.length) {
    const currentValue = isolatorInputType.value;
    isolatorInputType.innerHTML = "";
    payload.input_types.forEach((inputType) => {
      const option = document.createElement("option");
      option.value = inputType.id;
      option.textContent = inputType.label;
      isolatorInputType.appendChild(option);
    });
    isolatorInputType.value =
      payload.defaults?.input_type || currentValue || payload.input_types[0].id;
  }

  if (payload.defaults) {
    if (payload.defaults.strength) {
      isolatorStrength.value = String(payload.defaults.strength);
    }
    isolatorDeecho.value = payload.defaults.deecho ? "true" : "false";
    isolatorWidthFocus.value = payload.defaults.width_focus ? "true" : "false";
    if (typeof payload.defaults.clarity_preserve !== "undefined") {
      isolatorClarityPreserve.value = String(payload.defaults.clarity_preserve);
    }
  }

  updateSliderLabels();
  updateIsolatorModeSummary();
  updateIsolatorInputTypeSummary();
}

async function loadMasteringOptions() {
  const response = await fetch("/api/mastering/options");
  const payload = await response.json();
  state.masteringOptions = payload;
  if (payload?.defaults && typeof payload.defaults.resolution !== "undefined") {
    masteringResolution.value = String(payload.defaults.resolution);
  }
  if (payload?.limits) {
    if (typeof payload.limits.min_resolution !== "undefined") {
      masteringResolution.min = String(payload.limits.min_resolution);
    }
    if (typeof payload.limits.max_resolution !== "undefined") {
      masteringResolution.max = String(payload.limits.max_resolution);
    }
  }
  updateSliderLabels();
}

async function loadOptimizeOptions() {
  const response = await fetch("/api/optimize/options");
  const payload = await response.json();
  state.optimizeOptions = payload;
  if (payload?.defaults && typeof payload.defaults.max_cut_db !== "undefined") {
    optimizeStrength.value = String(payload.defaults.max_cut_db);
  } else if (payload?.defaults && typeof payload.defaults.stitch_strength !== "undefined") {
    const legacyStrength = Math.max(1, Math.min(20, Number(payload.defaults.stitch_strength)));
    const mappedDb = -38 + ((legacyStrength - 1) / 19) * 26;
    optimizeStrength.value = String(mappedDb.toFixed(1));
  } else if (payload?.defaults && typeof payload.defaults.isolate_strength !== "undefined") {
    optimizeStrength.value = String(payload.defaults.isolate_strength);
  }
  updateSliderLabels();
}

async function loadAlbumOptions() {
  const response = await fetch("/api/albums/options");
  const payload = await response.json();
  state.albumOptions = payload;
}

async function loadAlbumProjects(loadCurrentProject = true) {
  const response = await fetch("/api/albums/projects");
  const payload = await response.json();
  state.albumProjects = Array.isArray(payload.projects) ? payload.projects : [];
  renderAlbumProjectSelector();

  if (!loadCurrentProject) {
    return;
  }

  if (!state.albumProjects.length) {
    state.currentAlbumProjectId = null;
    renderAlbumProject(null);
    return;
  }

  const exists = state.albumProjects.some(
    (project) => project.id === state.currentAlbumProjectId,
  );
  if (!exists) {
    state.currentAlbumProjectId = state.albumProjects[0].id;
  }
  await loadAlbumProject(state.currentAlbumProjectId);
}

async function loadAlbumProject(projectId) {
  if (!projectId) {
    renderAlbumProject(null);
    return;
  }
  const response = await fetch(`/api/albums/projects/${projectId}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Could not load album project.");
  }
  state.currentAlbumProjectId = projectId;
  renderAlbumProject(payload.project || null);
  renderAlbumProjectSelector();
}

async function loadApiComposeOptions() {
  const response = await fetch("/api/api-compose/options");
  const payload = await response.json();
  state.apiComposeOptions = payload;
  if (payload?.defaults && payload.defaults.endpoint_url && apiComposeEndpointUrl) {
    apiComposeEndpointUrl.value = payload.defaults.endpoint_url;
  }
  if (payload?.defaults && payload.defaults.auth_header && apiComposeAuthHeader) {
    apiComposeAuthHeader.value = payload.defaults.auth_header;
  }
  if (payload?.defaults && typeof payload.defaults.extra_json === "string" && apiComposeExtraJson) {
    apiComposeExtraJson.value = payload.defaults.extra_json;
  }
}

async function loadTrainingOptions() {
  const response = await fetch("/api/training/options");
  const payload = await response.json();
  state.trainingOptions = payload;

  const messages = [];
  if (payload.warning) {
    messages.push(`<strong>Heads up</strong><p>${payload.warning}</p>`);
  }
  messages.push("<p><strong>Training modes</strong>: classic RVC + SUNO audition keeps `BASE*` as the real target-voice truth and uses `SUNO*` only for fixed checkpoint listening. Paired aligned conversion still uses `BASE*`, `TARGET*`, and `SUNO*` pairs. Concert remaster uses matched `CONCERT*` and `CD*` clips. Lyrics and persona plans are not used on this screen.</p>");
  messages.push("<p>Checkpoint previews now render a fixed 10 second audition clip from your uploaded SUNO set whenever a new saved checkpoint appears, so we can judge the voice by ear instead of by loss alone.</p>");

  if (messages.length) {
    trainingWarning.classList.remove("hidden");
    trainingWarning.innerHTML = messages.join("");
  } else {
    trainingWarning.classList.add("hidden");
    trainingWarning.innerHTML = "";
  }
  if (trainingOutputMode) {
    const outputModes = Array.isArray(payload?.output_modes) ? payload.output_modes : [];
    trainingOutputMode.innerHTML = outputModes.map((option) => (
      `<option value="${escapeHtml(option.id || "")}">${escapeHtml(option.label || option.id || "Unknown mode")}</option>`
    )).join("");
    const defaultMode = String(payload?.defaults?.output_mode || "classic-rvc-support").trim().toLowerCase();
    trainingOutputMode.value = Array.from(trainingOutputMode.options).some((option) => option.value === defaultMode)
      ? defaultMode
      : (trainingOutputMode.options[0]?.value || "classic-rvc-support");
  }
  if (trainingVersion) {
    trainingVersion.innerHTML = '<option value="v2">Persona v1.1 / v1.0 backbone (v2)</option>';
    trainingVersion.value = "v2";
  }
  if (trainingEpochMode) {
    trainingEpochMode.value = "fixed";
  }
  syncTrainingRunModeUI();
}

async function loadDetagOptions() {
  const response = await fetch("/api/detag/options");
  const payload = await response.json();
  state.detagVoices = payload.voices || [];
  renderDetagVoices();
}

function getTransposeValue() {
  if (pitchPreset.value === "custom") {
    return Number(customPitch.value || 0);
  }
  return Number(pitchPreset.value);
}

function getPreviewSourceFile() {
  return state.files[0] || null;
}

function getPreviewSourceKey(file) {
  if (!file) {
    return "";
  }
  return `${file.name}:${file.size}:${file.lastModified || 0}`;
}

function invalidatePreviewSource() {
  state.previewSourceId = null;
  state.previewSourceKey = "";
}

function resetPreviewState(title = "Waiting for a file", message = "Add a file to hear the auto preview.") {
  if (state.previewDebounceHandle) {
    clearTimeout(state.previewDebounceHandle);
    state.previewDebounceHandle = null;
  }
  previewMeta.classList.add("hidden");
  previewMeta.innerHTML = "";
  previewPlayer.pause();
  previewPlayer.removeAttribute("src");
  previewPlayer.load();
  previewPlayer.classList.add("hidden");
  setManagedAudioVisible(previewPlayer, false);
  setPreviewStatus("idle", title, message, 0);
}

async function ensurePreviewSourceUploaded() {
  const sourceFile = getPreviewSourceFile();
  if (!sourceFile) {
    throw new Error("Add a file before requesting a preview.");
  }

  const sourceKey = getPreviewSourceKey(sourceFile);
  if (state.previewSourceId && state.previewSourceKey === sourceKey) {
    return state.previewSourceId;
  }

  const data = new FormData();
  data.append("file", sourceFile, sourceFile.name);
  const response = await fetch("/api/preview/source", {
    method: "POST",
    body: data,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Could not upload the preview source.");
  }

  state.previewSourceId = payload.source_id;
  state.previewSourceKey = sourceKey;
  return state.previewSourceId;
}

function scheduleAutoPreview() {
  resetPreviewState("Preview disabled", "Lead Builder now runs only the full direct-conversion path instead of the old 5 second preview.");
}

async function generateAutoPreview(requestToken) {
  resetPreviewState("Preview disabled", "Lead Builder now runs only the full direct-conversion path instead of the old 5 second preview.");
  void requestToken;
  return;
  try {
    if (outputMode && outputMode.value === "blend" && !secondaryModelSelect.value) {
      throw new Error("Pick a second model before generating a blend preview.");
    }
    setPreviewStatus("running", "Uploading preview source", "Caching the preview source file...", 18);
    const sourceId = await ensurePreviewSourceUploaded();
    if (requestToken !== state.previewRequestToken) {
      return;
    }

    const data = new FormData();
    data.append("source_id", sourceId);
    data.append("model_name", modelSelect.value);
    data.append("output_mode", outputMode?.value || "single");
    data.append("secondary_model_name", secondaryModelSelect?.value || "");
    data.append("blend_percentage", blendPercentage?.value || "50");
    data.append("transpose", String(getTransposeValue()));
    data.append("quality_preset", qualityPreset.value);
    data.append("preprocess_mode", preprocessMode.value);
    data.append("preprocess_strength", preprocessStrength.value);
    data.append("pitch_method", pitchMethod.value);
    data.append("index_path", indexPath.value);
    data.append("index_rate", String(Number(indexRate.value) / 100));
    data.append("protect", String(Number(protect.value) / 100));
    data.append("rms_mix_rate", String(Number(rmsMixRate.value) / 100));
    data.append("filter_radius", String(Number(filterRadius.value)));
    if (state.indexFile) {
      data.append("index_file", state.indexFile, state.indexFile.name);
    }

    setPreviewStatus("running", "Generating preview", "Converting the middle 5 second snippet with your current settings...", 42);
    const response = await fetch("/api/preview", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (requestToken !== state.previewRequestToken) {
      return;
    }
    if (!response.ok) {
      throw new Error(payload.detail || "Could not generate the preview.");
    }

    const previewUrl = `${payload.preview_url}?t=${Date.now()}`;
    const previewPipelineId = payload.preprocess_mode || payload.cleanup_mode || "off";
    const previewPipelineLabel = payload.preprocess_label || getPreprocessLabel(previewPipelineId);
    setManagedAudioMeta(
      previewPlayer,
      getPreviewSourceFile()?.name || "Preview",
      `5 second middle snippet | Pipeline: ${previewPipelineLabel}`,
    );
    previewPlayer.src = previewUrl;
    previewPlayer.classList.remove("hidden");
    setManagedAudioVisible(previewPlayer, true);
    previewMeta.classList.remove("hidden");
    previewMeta.innerHTML = `
      <strong>Latest preview ready</strong>
      <p>Using the first selected file: ${getPreviewSourceFile()?.name || "unknown file"}. Snippet starts around ${payload.clip_start}s and lasts ${payload.clip_duration}s. Pipeline: ${previewPipelineLabel}.${payload.output_mode === "blend" ? ` Blend: ${payload.blend_percentage}% primary / ${100 - Number(payload.blend_percentage || 0)}% ${payload.secondary_model_name || "secondary"}.` : ""}</p>
    `;
    setPreviewStatus("completed", "Preview ready", "Listen here before running the full conversion.", 100);
  } catch (error) {
    if (requestToken !== state.previewRequestToken) {
      return;
    }
    previewMeta.classList.add("hidden");
    previewMeta.innerHTML = "";
    previewPlayer.classList.add("hidden");
    setManagedAudioVisible(previewPlayer, false);
    previewPlayer.removeAttribute("src");
    previewPlayer.load();
    setPreviewStatus(
      "failed",
      "Preview failed",
      error.message || String(error),
      0,
    );
  }
}

function renderResults(job) {
  const artifacts = [
    {
      title: job.final_download_name || "master_conversion.wav",
      meta: "Final target lead",
      url: job.final_url,
      downloadName: job.final_download_name || "master_conversion.wav",
    },
    {
      title: job.metadata_download_name || "master_conversion_report.json",
      meta: "Similarity, gap-fill, and timing report",
      url: job.metadata_url,
      downloadName: job.metadata_download_name || "master_conversion_report.json",
      audio: false,
    },
  ].filter((entry) => entry.url);

  resultSummary.classList.toggle("hidden", !(job.status === "completed" && artifacts.length));
  if (job.status === "completed" && artifacts.length) {
    const zipLink = job.zip_url
      ? `<a class="zip-link" href="${job.zip_url}" download>Download full master-conversion package</a>`
      : "";
    const attempts = Number(job.repair_attempts || 0);
    const rebuilt = Number(job.repaired_word_count || 0);
    const directOnly = Boolean(job.direct_conversion_only);
    const processLine = directOnly
      ? "Direct model render only. No preprocessing, repair, or final cleanup was applied."
      : (
        attempts === 0 && rebuilt === 0
          ? "Full blueprint conversion was used for the whole lead."
          : `Weak regions tested: ${attempts}. Phrase gaps rebuilt: ${rebuilt}.`
      );
    const similarityLabel = directOnly
      ? "Lyric similarity report on the direct render"
      : "Best lyric similarity after cleanup";
    resultSummary.innerHTML = `
      <strong>Master Conversion finished</strong>
      <p>Source: ${job.source_name || "unknown song"}. Voice: ${job.model_name || "unknown model"}.${job.output_mode === "blend" ? ` Blend: ${Number(job.blend_percentage || 0)}% primary / ${100 - Number(job.blend_percentage || 0)}% ${job.secondary_model_name || "secondary"}.` : ""}</p>
      <p>${similarityLabel}: ${Number(job.best_similarity_score || 0).toFixed(2)}%. ${processLine}</p>
      <p>Timing: npy ${Number(job.timings?.npy || 0).toFixed(2)}s, f0 ${Number(job.timings?.f0 || 0).toFixed(2)}s, infer ${Number(job.timings?.infer || 0).toFixed(2)}s. Sample rate: ${Number(job.sample_rate || 0)} Hz.</p>
      ${job.best_word_report ? `<p>${job.best_word_report}</p>` : ""}
      ${job.best_letter_report ? `<p>${job.best_letter_report}</p>` : ""}
      ${zipLink}
    `;
  } else {
    resultSummary.innerHTML = "";
  }

  results.innerHTML = "";
  artifacts.forEach((artifact) => {
    const card = document.createElement("div");
    card.className = "result-row";
    card.innerHTML = `
      <div class="result-name">${artifact.title}</div>
      <div class="result-meta">${artifact.meta}</div>
      ${artifact.audio === false ? "" : buildManagedAudioMarkup(artifact.url, artifact.title, artifact.meta)}
      <div class="result-links">
        <a class="result-link" href="${artifact.url}" download="${artifact.downloadName}">${artifact.audio === false ? "Download report" : "Download file"}</a>
      </div>
    `;
    results.appendChild(card);
  });
  hydrateManagedAudio(results);
}

function renderGenerateResult(job) {
  generateResults.innerHTML = "";
  if (job.status !== "completed" || !job.result_url) {
    generateResultSummary.classList.add("hidden");
    generateResultSummary.innerHTML = "";
    return;
  }

  generateResultSummary.classList.remove("hidden");
  const bpmLine =
    Number(job.guide_bpm || 0) > 0 && Number(job.target_bpm || 0) > 0
      ? `Guide BPM ${Number(job.guide_bpm).toFixed(2)} -> Target BPM ${Number(job.target_bpm).toFixed(2)}${job.tempo_adjusted ? ` (tempo ratio ${Number(job.tempo_ratio || 1).toFixed(3)})` : ""}.`
      : "No BPM conform applied.";
  const keyLine =
    job.guide_key && job.target_key
      ? `Guide key ${job.guide_key} -> Target key ${job.target_key} (${Number(job.transpose || 0)} semitones).`
      : "No key transpose applied.";
  const preprocessLabel = getPreprocessLabel(job.preprocess_mode || "off");
  const similarity = Number(job.best_similarity_score || 0);
  const attempts = Number(job.repair_attempts || 0);
  const repairedWords = Number(job.repaired_word_count || 0);
  const regenerationLine = job.regeneration_available
    ? "XTTS word regeneration was available for heavier repair passes."
    : (job.regeneration_reason || "Local repair passes ran without XTTS regeneration.");
  generateResultSummary.innerHTML = `
    <strong>Pronunciation-repaired vocal ready</strong>
    <p>The reference vocal was converted into ${job.model_name}, then repaired against the pasted lyrics.</p>
    <p>Best ordered-word similarity: ${similarity.toFixed(2)}%. Repair attempts: ${attempts}. Repaired word regions: ${repairedWords}.</p>
    <p>${keyLine} ${bpmLine}</p>
    <p>Pipeline: ${preprocessLabel}. Sample rate: ${Number(job.sample_rate || 0)} Hz.</p>
    ${job.best_word_report ? `<p>${job.best_word_report}</p>` : ""}
    ${job.best_letter_report ? `<p>${job.best_letter_report}</p>` : ""}
    <p>${regenerationLine}</p>
  `;

  const cards = [
    {
      title: job.download_name || "generated_vocal.wav",
      meta: `Repaired output | Reference: ${job.guide_name || "unknown"} | Voice: ${job.model_name}`,
      url: job.result_url,
      downloadName: job.download_name || "generated_vocal.wav",
      audio: true,
    },
    {
      title: job.repair_source_download_name || "converted_before_repair.wav",
      meta: "Converted voice before pronunciation repair",
      url: job.repair_source_url,
      downloadName: job.repair_source_download_name || "converted_before_repair.wav",
      audio: true,
    },
    {
      title: job.metadata_download_name || "generation_metadata.json",
      meta: job.lyrics_preview
        ? `Stored lyrics preview: ${job.lyrics_preview}`
        : "Stored generation metadata and lyrics",
      url: job.metadata_url,
      downloadName: job.metadata_download_name || "generation_metadata.json",
      audio: false,
    },
  ];

  cards.forEach((entry) => {
    if (!entry.url) {
      return;
    }
    const card = document.createElement("div");
    card.className = "result-row";
    card.innerHTML = `
      <div class="result-name">${entry.title}</div>
      <div class="result-meta">${entry.meta}</div>
      ${entry.audio ? buildManagedAudioMarkup(entry.url, entry.title, entry.meta) : ""}
      <div class="result-links">
        <a class="result-link" href="${entry.url}" download="${entry.downloadName}">${entry.audio ? "Download vocal" : "Download metadata"}</a>
      </div>
    `;
    generateResults.appendChild(card);
  });
  hydrateManagedAudio(generateResults);
}

function renderTrainingResult(job) {
  if (job.status === "completed") {
    trainingResultSummary.classList.remove("hidden");
    const selectedOutputMode = String(job.output_mode || "").trim().toLowerCase();
    const modeConfig = getTrainingModeConfig(selectedOutputMode);
    const isLogicOnly = isPersonaPackageMode(job.output_mode) || job.output_mode === "pipa-logic-only";
    const directPairMode = selectedOutputMode === "persona-aligned-pth" || selectedOutputMode === "concert-remaster-paired";
    const classicSupportMode = selectedOutputMode === "classic-rvc-support";
    const indexBlock = job.index_path
      ? `<div class="result-meta">Index: ${job.index_path}</div>`
      : `<div class="result-meta">Index: not created (${isLogicOnly ? `${getPersonaPackageLabel(job.output_mode)} does not use a legacy index` : (job.build_index ? "full bundle did not produce one" : "lite mode")})</div>`;
    const outputModeLabel = classicSupportMode
      ? modeConfig.label
      : directPairMode
      ? modeConfig.label
      : (isLogicOnly
      ? getPersonaPackageLabel(job.output_mode)
      : (job.output_mode === "pipa-lite" ? "PIPA lite" : "PIPA full"));
    const outputModeBlock = `<div class="result-meta">Output: ${outputModeLabel}</div>`;
    const transcriptBlock = classicSupportMode
      ? `<div class="result-meta">Dataset coverage: ${Number(job.base_voice_clip_count || 0)} BASE truth clips, ${Number(job.paired_song_count || 0)} extra target truth clips, ${Number(job.depersonafied_variant_count || 0)} SUNO audition clips, ${Number(job.skipped_audio_files || 0)} ignored.</div>`
      : directPairMode
      ? `<div class="result-meta">Alignment coverage: ${Number(job.matched_audio_files || 0)} matched ${modeConfig.targetName}/${modeConfig.sourceName} pairs across ${Number(job.total_audio_files || 0)} uploaded clips. Unused clips: ${Number(job.skipped_audio_files || 0)}.</div>`
      : `<div class="result-meta">Transcript coverage: ${Number(job.matched_audio_files || 0)}/${Number(job.total_audio_files || 0)} matched, ${Number(job.skipped_audio_files || 0)} skipped.</div>`;
    const referenceBlock = classicSupportMode
      ? ""
      : `<div class="result-meta">Reference bank: ${Number(job.reference_word_count || 0)} word refs, ${Number(job.reference_phrase_count || 0)} phrase refs.</div>`;
    const recipeParts = [];
    if (Number(job.base_voice_clip_count || 0)) {
      recipeParts.push(`${Number(job.base_voice_clip_count || 0)} ${modeConfig.baseName || "BASE"} clips`);
    }
    if (Number(job.paired_song_count || 0)) {
      recipeParts.push(
        classicSupportMode
          ? `${Number(job.paired_song_count || 0)} ${modeConfig.targetName} truth clips`
          : `${Number(job.paired_song_count || 0)} ${modeConfig.targetName}/${modeConfig.sourceName} pairs`,
      );
    }
    if (Number(job.depersonafied_variant_count || 0)) {
      recipeParts.push(
        classicSupportMode
          ? `${Number(job.depersonafied_variant_count || 0)} ${modeConfig.sourceName} audition clips`
          : `${Number(job.depersonafied_variant_count || 0)} aligned ${modeConfig.sourceName.toLowerCase()} sources`,
      );
    }
    const recipeBlock = recipeParts.length ? `<div class="result-meta">Recipe: ${recipeParts.join(" | ")}</div>` : "";
    const planBlock = job.training_plan_path
      ? `<div class="result-meta">Persona plan: ${job.training_plan_path}</div>`
      : "";
    const selectionBlock = job.pipa_selection_name
      ? `<div class="result-meta">${isLogicOnly ? "Persona package selection" : "One-click model selection"}: ${job.pipa_selection_name}</div>`
      : "";
    const manifestBlock = job.pipa_manifest_path
      ? `<div class="result-meta">PIPA manifest: ${job.pipa_manifest_path}</div>`
      : "";
    const profileBlock = job.phoneme_profile_path
      ? `<div class="result-meta">Pronunciation profile: ${job.phoneme_profile_path}</div>`
      : "";
    const guidedCheckpointBlock = job.guided_regeneration_path
      ? `<div class="result-meta">Pronunciation regenerator checkpoint: ${job.guided_regeneration_path}</div>`
      : `<div class="result-meta">Pronunciation regenerator checkpoint: not written</div>`;
    const guidedConfigBlock = job.guided_regeneration_config_path
      ? `<div class="result-meta">Pronunciation regenerator config: ${job.guided_regeneration_config_path}</div>`
      : "";
    const guidedReportBlock = job.guided_regeneration_report_path
      ? `<div class="result-meta">Pronunciation regenerator report: ${job.guided_regeneration_report_path}</div>`
      : "";
    const guidedPreviewBlock = job.guided_regeneration_preview_path
      ? `<div class="result-meta">Pronunciation preview render: ${job.guided_regeneration_preview_path}</div>`
      : "";
    const guidedTargetPreviewBlock = job.guided_regeneration_target_preview_path
      ? `<div class="result-meta">Pronunciation target preview: ${job.guided_regeneration_target_preview_path}</div>`
      : "";
    const guidedMetricsBlock = job.guided_regeneration_path
      ? `<div class="result-meta">Voice-builder fit: best epoch ${Number(job.guided_regeneration_best_epoch || 0)} | best total ${Number(job.guided_regeneration_best_val_total || 0).toFixed(4)} | best quality ${Number((job.guided_regeneration_best_quality ?? job.guided_regeneration_best_target_quality) || 0).toFixed(4)} | best mel ${Number(job.guided_regeneration_best_val_l1 || 0).toFixed(4)} | lyric phone acc ${(Number(job.guided_regeneration_best_lyric_phone_accuracy || 0) * 100).toFixed(1)}% | voicing ${(Number(job.guided_regeneration_best_vuv_accuracy || 0) * 100).toFixed(1)}% | plateau ${Number(job.guided_regeneration_plateau_epochs || 0)} epochs | training slices ${Number(job.guided_regeneration_sample_count || 0)}</div>`
      : "";
    const guidedHardwareBlock = job.guided_regeneration_hardware_summary
      ? `<div class="result-meta">Training hardware: ${job.guided_regeneration_hardware_summary}</div>`
      : "";
    const guidedQualityBlock = job.guided_regeneration_quality_summary
      ? `<div class="result-meta">Last training state: ${job.guided_regeneration_quality_summary}</div>`
      : "";
    const rebuildProfileBlock = job.rebuild_profile_path
      ? `<div class="result-meta">Rebuild profile: ${job.rebuild_profile_path}</div>`
      : "";
    const rebuildClipReportsBlock = job.rebuild_clip_reports_path
      ? `<div class="result-meta">Rebuild clip reports: ${job.rebuild_clip_reports_path}</div>`
      : "";
    const stopBlock = job.stopped_early
      ? `<div class="result-meta">Training stop: manually stopped after a saved checkpoint was available.</div>`
      : "";
    trainingResultSummary.innerHTML = `
      <strong>${job.experiment_name} PIPA package is ready</strong>
      <p>${
        classicSupportMode
          ? "This package is ready as a base-first classic RVC conversion voice. BASE clips were kept as the real speaker truth, and the same SUNO audition source was rendered at each checkpoint so you can judge how the model handles that source domain by ear."
          : directPairMode
          ? (selectedOutputMode === "concert-remaster-paired"
            ? "This package is ready for direct concert remaster conversion. It was trained from matched CONCERT/CD supervision, with extra detail-focused oversampling to push the model harder on the parts human ears notice first."
            : "This package is ready for direct paired full-vocal conversion. It was trained from BASE identity clips plus matched TARGET/SUNO supervision, with extra detail-focused oversampling to push the model harder on the parts human ears notice first.")
          : (isLogicOnly
            ? "Persona v1.0 trained the guide-conditioned voice-builder regenerator and pronunciation package without creating any legacy .pth backbone. This package is ready for the new conversion flow."
            : (job.stopped_early ? "The latest saved voice-builder and backbone checkpoints were packaged when you stopped training." : "Your trained PIPA package now includes both the guide-conditioned voice-builder regenerator and the RVC backbone:"))
      }</p>
      ${isLogicOnly ? "" : `<div class="result-meta">${job.model_path || "weights file not found"}</div>`}
      ${outputModeBlock}
      ${indexBlock}
      ${stopBlock}
      ${recipeBlock}
      ${planBlock}
      ${selectionBlock}
      ${manifestBlock}
      ${profileBlock}
      ${guidedCheckpointBlock}
      ${guidedConfigBlock}
      ${guidedReportBlock}
      ${guidedPreviewBlock}
      ${guidedTargetPreviewBlock}
      ${guidedMetricsBlock}
      ${guidedHardwareBlock}
      ${guidedQualityBlock}
      ${rebuildProfileBlock}
      ${rebuildClipReportsBlock}
      ${transcriptBlock}
      ${directPairMode ? "" : referenceBlock}
      <p>${
        classicSupportMode
          ? "Refresh Lead Builder to use this voice in the normal conversion flow. The latest checkpoint audition stays available below so you can compare what training actually sounded like."
          : directPairMode
          ? "Refresh Lead Builder to use this voice in the simplified direct conversion flow."
          : "This package now includes pronunciation references, guide-following rebuild metadata, and a direct vocal-regeneration checkpoint trained from pure-voice identity clips plus paired de-personafied to target song slices."
      }</p>
        <p>${
          directPairMode
            ? "Lead Builder now shows conversion voices, so this model should appear there after refresh."
            : "Refresh the Master Conversion voice package list to use it in one click."
        }</p>
      `;
  } else {
    trainingResultSummary.classList.add("hidden");
    trainingResultSummary.innerHTML = "";
  }
}

function renderTrainingCheckpointPreview(job) {
  if (!trainingCheckpointPreview || !trainingCheckpointPreviewMeta || !trainingCheckpointPreviewPlayer) {
    return;
  }
  const previewUrl = String(job?.checkpoint_preview_url || "").trim();
  if (!previewUrl) {
    trainingCheckpointPreview.classList.add("hidden");
    trainingCheckpointPreviewMeta.innerHTML = "";
    trainingCheckpointPreviewPlayer.pause();
    trainingCheckpointPreviewPlayer.removeAttribute("src");
    trainingCheckpointPreviewPlayer.load();
    setManagedAudioVisible(trainingCheckpointPreviewPlayer, false);
    return;
  }
  const epoch = Number(job?.checkpoint_preview_epoch || 0);
  const sourceName = job?.checkpoint_preview_source_name || "checkpoint audition source";
  const sourceRole = job?.checkpoint_preview_source_role || "SUNO";
  const downloadName = job?.checkpoint_preview_download_name || `epoch_${String(epoch).padStart(5, "0")}.wav`;
  const statusLine = job?.checkpoint_preview_status || `Latest checkpoint audition is ready from epoch ${epoch}.`;
  const meta = `Checkpoint audition | Epoch ${epoch || "?"} | ${sourceRole} source: ${sourceName}`;

  trainingCheckpointPreview.classList.remove("hidden");
  setManagedAudioMeta(trainingCheckpointPreviewPlayer, downloadName, meta);
  if ((trainingCheckpointPreviewPlayer.getAttribute("src") || "") !== previewUrl) {
    trainingCheckpointPreviewPlayer.pause();
    trainingCheckpointPreviewPlayer.currentTime = 0;
    trainingCheckpointPreviewPlayer.src = previewUrl;
    trainingCheckpointPreviewPlayer.load();
  }
  setManagedAudioVisible(trainingCheckpointPreviewPlayer, true);
  trainingCheckpointPreviewMeta.innerHTML = `
    <strong>Latest checkpoint audition</strong>
    <p>${escapeHtml(statusLine)}</p>
    <div class="result-links">
      <a class="result-link" href="${escapeHtml(previewUrl)}" download="${escapeHtml(downloadName)}">Download audition</a>
    </div>
  `;
}

function renderDetagResult(job) {
  detagResults.innerHTML = "";
  if (job.status !== "completed" || !job.result_url) {
    detagResultSummary.classList.add("hidden");
    detagResultSummary.innerHTML = "";
    return;
  }

  detagResultSummary.classList.remove("hidden");
  detagResultSummary.innerHTML = `
    <strong>Selected voice kept</strong>
    <p>Similarity threshold: ${job.threshold}. Kept audio ratio: ${Math.round((job.kept_ratio || 0) * 100)}%.</p>
  `;

  const card = document.createElement("div");
  card.className = "result-row";
  card.innerHTML = `
    <div class="result-name">${job.download_name || "Detag output"}</div>
    <div class="result-meta">Only the selected voice should remain in this file.</div>
    ${buildManagedAudioMarkup(job.result_url, job.download_name || "Detag output", "Selected voice only")}
    <div class="result-links">
      <a class="result-link" href="${job.result_url}" download="${job.download_name}">Download file</a>
    </div>
  `;
  detagResults.appendChild(card);
  hydrateManagedAudio(detagResults);
}

function renderIsolatorResult(job) {
  isolatorResults.innerHTML = "";
  if (job.status !== "completed" || !job.main_vocal_url || !job.backing_vocal_url) {
    isolatorResultSummary.classList.add("hidden");
    isolatorResultSummary.innerHTML = "";
    return;
  }

  isolatorResultSummary.classList.remove("hidden");
  const isReverbEchoOnly = job.mode === "reverb-echo-only";
  const sourceFiles = Array.isArray(job.source_files) ? job.source_files : [];
  const currentFile = job.current_file ? `Current file: ${job.current_file}. ` : "";
  const sourceSummary = sourceFiles.length
    ? `Source files: ${sourceFiles.join(", ")}. `
    : "";
  isolatorResultSummary.innerHTML = `
    <strong>Two stems ready</strong>
    <p>${currentFile}${sourceSummary}Input type: ${job.input_type}. Mode: ${job.mode}. Sample rate: ${job.sample_rate} Hz. Strength: ${job.strength}. De-echo: ${job.deecho ? "On" : "Off"}. Stereo-aware focus: ${job.width_focus ? "On" : "Off"}. Lead clarity preserve: ${job.clarity_preserve}%.</p>
  `;

  const stems = [
    {
      title: isReverbEchoOnly ? "Cleaned vocal" : "Main vocal",
      url: job.main_vocal_url,
      downloadName: job.main_vocal_download_name,
      meta: isReverbEchoOnly
        ? "This is the vocal after reverb and echo cleanup."
        : "This is the cleaned lead/main vocal stem.",
    },
    {
      title: isReverbEchoOnly ? "Removed reverb/echo layer" : "Backing vocal",
      url: job.backing_vocal_url,
      downloadName: job.backing_vocal_download_name,
      meta: isReverbEchoOnly
        ? "This contains the reverb/echo material that was removed."
        : "This is the backing/adlib layer that was split away from the lead.",
    },
  ];

  stems.forEach((stem) => {
    const card = document.createElement("div");
    card.className = "result-row";
    card.innerHTML = `
      <div class="result-name">${stem.title}</div>
      <div class="result-meta">${stem.meta}</div>
      ${buildManagedAudioMarkup(stem.url, stem.title, stem.meta)}
      <div class="result-links">
        <a class="result-link" href="${stem.url}" download="${stem.downloadName}">Download file</a>
      </div>
    `;
    isolatorResults.appendChild(card);
  });
  hydrateManagedAudio(isolatorResults);
}

function drawMasteringProfile(points) {
  masteringProfileChart.innerHTML = "";
  if (!Array.isArray(points) || !points.length) {
    masteringProfileCard.classList.add("hidden");
    masteringProfileMeta.textContent = "";
    return;
  }

  masteringProfileCard.classList.remove("hidden");
  const width = 640;
  const height = 240;
  const paddingX = 26;
  const paddingY = 20;
  const values = points.map((point) => Number(point.gain_db || 0));
  const limit = Math.max(3, Math.ceil(Math.max(...values.map((value) => Math.abs(value)))));

  const gridLevels = [-limit, 0, limit];
  gridLevels.forEach((level) => {
    const y =
      paddingY +
      ((limit - level) / (limit * 2 || 1)) * (height - paddingY * 2);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", "0");
    line.setAttribute("x2", String(width));
    line.setAttribute("y1", String(y));
    line.setAttribute("y2", String(y));
    line.setAttribute("stroke", level === 0 ? "rgba(29, 185, 84, 0.72)" : "rgba(29, 185, 84, 0.14)");
    line.setAttribute("stroke-width", level === 0 ? "2" : "1");
    masteringProfileChart.appendChild(line);
  });

  const pathParts = points.map((point, index) => {
    const x = paddingX + (index / Math.max(points.length - 1, 1)) * (width - paddingX * 2);
    const normalized = (limit - Number(point.gain_db || 0)) / (limit * 2 || 1);
    const y = paddingY + normalized * (height - paddingY * 2);
    return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  });
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", pathParts.join(" "));
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "#1db954");
  path.setAttribute("stroke-width", "4");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  masteringProfileChart.appendChild(path);
}

function renderMasteringResult(job) {
  masteringResults.innerHTML = "";
  if (job.status !== "completed" || !job.mastered_url) {
    masteringResultSummary.classList.add("hidden");
    masteringResultSummary.innerHTML = "";
    masteringProfileCard.classList.add("hidden");
    masteringProfileChart.innerHTML = "";
    masteringProfileMeta.textContent = "";
    return;
  }

  masteringResultSummary.classList.remove("hidden");
  masteringResultSummary.innerHTML = `
    <strong>EQ profile matched</strong>
    <p>Resolution: ${job.resolution}. Sample rate: ${job.sample_rate} Hz. Reference masters blended: ${Number(job.reference_count || 0)}. Loudness shift: ${Number(job.loudness_gain_db || 0).toFixed(1)} dB. Source RMS: ${Number(job.source_rms_db || 0).toFixed(1)} dB. Reference RMS: ${Number(job.reference_rms_db || 0).toFixed(1)} dB.</p>
  `;

  const bands = job.band_summary || {};
  masteringProfileMeta.textContent = `Low ${Number(bands.low || 0).toFixed(1)} dB | Mid ${Number(bands.mid || 0).toFixed(1)} dB | Air ${Number(bands.air || 0).toFixed(1)} dB`;
  drawMasteringProfile(job.curve_points || []);

  const cards = [
    {
      title: job.mastered_download_name || "Matched master",
      meta: "This is your source file after its EQ balance was matched toward the blended mastered reference profile.",
      url: job.mastered_url,
      downloadName: job.mastered_download_name,
      audio: true,
    },
    {
      title: job.profile_download_name || "EQ profile JSON",
      meta: "This contains the learned EQ profile points and loudness metadata.",
      url: job.profile_url,
      downloadName: job.profile_download_name,
      audio: false,
    },
  ];

  cards.forEach((entry) => {
    const card = document.createElement("div");
    card.className = "result-row";
    card.innerHTML = `
      <div class="result-name">${entry.title}</div>
      <div class="result-meta">${entry.meta}</div>
      ${entry.audio ? buildManagedAudioMarkup(entry.url, entry.title, entry.meta) : ""}
      <div class="result-links">
        <a class="result-link" href="${entry.url}" download="${entry.downloadName}">${entry.audio ? "Download mastered file" : "Download EQ profile"}</a>
      </div>
    `;
    masteringResults.appendChild(card);
  });
  hydrateManagedAudio(masteringResults);
}

function renderOptimizeResult(job) {
  optimizeResults.innerHTML = "";
  const rankings = Array.isArray(job.rankings) ? job.rankings : [];
  const hasStitched = job.status === "completed" && Boolean(job.stitched_url);
  if (!hasStitched && !rankings.length) {
    optimizeResultSummary.classList.add("hidden");
    optimizeResultSummary.innerHTML = "";
    return;
  }

  optimizeResultSummary.classList.remove("hidden");
  const top = rankings[0];
  const topLine = top
    ? `Top take score: ${top.source_name} (${Number(top.score || 0).toFixed(1)}%).`
    : "";
  const seamLine = `Cut gate: ${Number(job.max_cut_db ?? -24).toFixed(0)} dB (skipped ${Number(job.skipped_by_db_gate || 0)} seam candidates).`;
  optimizeResultSummary.innerHTML = `
    <strong>Stitched acapella ready</strong>
    <p>${topLine} Replaced ${Number(job.replaced_word_count || 0)} / ${Number(job.total_word_count || 0)} lyric words. Anchor take: ${job.anchor_source_name || "n/a"}. ${seamLine}</p>
  `;

  if (job.stitched_url) {
    const stitchedCard = document.createElement("div");
    stitchedCard.className = "result-row";
    stitchedCard.innerHTML = `
      <div class="result-name">${job.stitched_download_name || "stitched_best_acapella.wav"}</div>
      <div class="result-meta">This is the final best-part comp generated from all uploaded takes.</div>
      ${buildManagedAudioMarkup(job.stitched_url, job.stitched_download_name || "Stitched acapella", "Best-part comp")}
      <div class="result-links">
        <a class="result-link" href="${job.stitched_url}" download="${job.stitched_download_name || "stitched_best_acapella.wav"}">Download stitched acapella</a>
      </div>
    `;
    optimizeResults.appendChild(stitchedCard);
  }

  const edits = Array.isArray(job.edits_preview) ? job.edits_preview : [];
  if (edits.length) {
    const editsCard = document.createElement("div");
    editsCard.className = "result-row";
    editsCard.innerHTML = `
      <div class="result-name">Top replacements</div>
      <div class="result-meta">${edits
        .slice(0, 10)
        .map((entry) => `${entry.word} (${Number(entry.before || 0).toFixed(0)}% → ${Number(entry.after || 0).toFixed(0)}%)`)
        .join(" | ")}</div>
    `;
    optimizeResults.appendChild(editsCard);
  }

  rankings.forEach((entry) => {
    const issueLines = Array.isArray(entry.issues) ? entry.issues : [];
    const issueText = issueLines.length ? issueLines.join(" | ") : "No major red flags detected.";
    const card = document.createElement("div");
    card.className = "result-row";
    card.innerHTML = `
      <div class="result-name">#${entry.rank} ${entry.source_name}</div>
      <div class="result-meta">Lyric intelligibility: ${Number(entry.score || 0).toFixed(1)}%</div>
      <div class="result-meta">${entry.summary || ""}</div>
      <div class="result-meta">Weakest words: ${entry.weak_words_summary || issueText}</div>
      ${buildManagedAudioMarkup(entry.prepared_vocal_url, entry.source_name, "Prepared vocal take")}
      <div class="result-links">
        <a class="result-link" href="${entry.prepared_vocal_url}" download="${entry.prepared_vocal_download_name || "prepared_vocal.wav"}">Download take</a>
      </div>
    `;
    optimizeResults.appendChild(card);
  });
  hydrateManagedAudio(optimizeResults);
}

function renderApiComposeResult(job) {
  apiComposeResultSummary.classList.add("hidden");
  apiComposeResultSummary.innerHTML = "";
  apiComposeResults.innerHTML = "";
  apiComposeResponse.textContent = "ACE-Step response will appear here.";

  if (job.status !== "completed" && job.status !== "failed") {
    return;
  }

  const responseText = (job.response_preview || "").trim();
  if (responseText) {
    apiComposeResponse.textContent = responseText;
  } else if (job.error) {
    apiComposeResponse.textContent = job.error;
  }

  apiComposeResultSummary.classList.remove("hidden");
  const taskIdLabel = job.task_id ? ` Task: ${job.task_id}.` : "";
  apiComposeResultSummary.innerHTML = `
    <strong>ACE-Step request ${job.status === "completed" ? "finished" : "failed"}</strong>
    <p>Endpoint: ${job.endpoint_url || "unknown"}. HTTP status: ${Number(job.provider_status_code || 0) || "n/a"}.${taskIdLabel}</p>
    <p>MIDI: ${job.midi_name || "n/a"}. Beat: ${job.beat_name || "none"}.</p>
  `;

  const audioUrls = Array.isArray(job.audio_urls) ? job.audio_urls : [];
  audioUrls.forEach((url, index) => {
    const card = document.createElement("div");
    card.className = "result-row";
    card.innerHTML = `
      <div class="result-name">ACE-Step Output ${index + 1}</div>
      ${buildManagedAudioMarkup(url, `ACE-Step Output ${index + 1}`, "Generated output")}
      <div class="result-links">
        <a class="result-link" href="${url}" target="_blank" rel="noopener">Open audio</a>
      </div>
    `;
    apiComposeResults.appendChild(card);
  });
  hydrateManagedAudio(apiComposeResults);
}

function renderTouchUpResult(job) {
  touchUpResults.innerHTML = "";
  if (job.status !== "completed" || !job.result_url) {
    touchUpResultSummary.classList.add("hidden");
    touchUpResultSummary.innerHTML = "";
    return;
  }

  touchUpResultSummary.classList.remove("hidden");
  const weakWords = Array.isArray(job.best_word_scores)
    ? [...job.best_word_scores]
        .sort((left, right) => Number(left.similarity || 0) - Number(right.similarity || 0))
        .slice(0, 8)
        .map((entry) => `${entry.word} (${Number(entry.similarity || 0).toFixed(0)}%)`)
        .join(", ")
    : "";
  const weakLetters = Array.isArray(job.best_letter_scores)
    ? [...job.best_letter_scores]
        .sort((left, right) => Number(left.similarity || 0) - Number(right.similarity || 0))
        .slice(0, 12)
        .map((entry) => `${entry.letter} in ${entry.word} (${Number(entry.similarity || 0).toFixed(0)}%)`)
        .join(", ")
    : "";
  const regenStatus = job.regeneration_available
    ? "Word regeneration backend is available."
    : (job.regeneration_reason || "Word regeneration backend is not ready.");
  if (job.mode === "smart-removal") {
    touchUpResultSummary.innerHTML = `
      <strong>Smart removal finished</strong>
      <p>Best ordered-word similarity: ${Number(job.best_similarity_score || 0).toFixed(2)}%.</p>
      <p>Expected lyrics: ${job.source_word}.</p>
      <p>${job.best_word_report || "No word-confidence summary returned."}</p>
      <p>${job.best_letter_report || "No letter-confidence summary returned."}</p>
      <p>Kept segments: ${Number(job.kept_segment_count || 0)}. Kept duration: ${Number(job.kept_duration_seconds || 0).toFixed(2)}s. Removed duration: ${Number(job.removed_duration_seconds || 0).toFixed(2)}s.</p>
      <p>Strength: ${job.strength}. Output sample rate: ${job.sample_rate} Hz. Source RMS: ${Number(job.source_rms_db || 0).toFixed(1)} dB. Output RMS: ${Number(job.output_rms_db || 0).toFixed(1)} dB.</p>
    `;

    const keptCard = document.createElement("div");
    keptCard.className = "result-row";
    keptCard.innerHTML = `
      <div class="result-name">${job.download_name || "smart_removed.wav"}</div>
      <div class="result-meta">Lyric-aligned vocal kept after muting non-lyric sections.</div>
      ${buildManagedAudioMarkup(job.result_url, job.download_name || "smart_removed.wav", "Smart removal result")}
      <div class="result-links">
        <a class="result-link" href="${job.result_url}" download="${job.download_name || "smart_removed.wav"}">Download kept vocal</a>
      </div>
    `;
    touchUpResults.appendChild(keptCard);

    if (job.removed_url) {
      const removedCard = document.createElement("div");
      removedCard.className = "result-row";
      removedCard.innerHTML = `
        <div class="result-name">${job.removed_download_name || "removed_layer.wav"}</div>
        <div class="result-meta">Everything that got cut between lyric windows.</div>
        ${buildManagedAudioMarkup(job.removed_url, job.removed_download_name || "removed_layer.wav", "Removed layer")}
        <div class="result-links">
          <a class="result-link" href="${job.removed_url}" download="${job.removed_download_name || "removed_layer.wav"}">Download removed layer</a>
        </div>
      `;
      touchUpResults.appendChild(removedCard);
    }

    hydrateManagedAudio(touchUpResults);
    return;
  }
  touchUpResultSummary.innerHTML = `
    <strong>Weak-word detection finished</strong>
    <p>Best ordered-word similarity: ${Number(job.best_similarity_score || 0).toFixed(2)}%. Repair attempts: ${Number(job.repair_attempts || job.variants_tested || 0)}. Regions processed: ${Number(job.batch_index || 0)}.</p>
    <p>Expected lyrics: ${job.source_word}.</p>
    <p>${job.best_word_report || "No word-confidence summary returned."}</p>
    <p>${job.best_letter_report || "No letter-confidence summary returned."}</p>
    <p>${weakWords ? `Weakest ordered words: ${weakWords}.` : "No weak-word breakdown returned."}</p>
    <p>${weakLetters ? `Weakest letters: ${weakLetters}.` : "No weak-letter breakdown returned."}</p>
    <p>${regenStatus} ${job.detected_only ? "This run stopped after detection because regeneration was not enabled." : `Repaired words accepted: ${Number(job.repaired_word_count || 0)}.`}</p>
    <p>Strength: ${job.strength}. Output sample rate: ${job.sample_rate} Hz. Source RMS: ${Number(job.source_rms_db || 0).toFixed(1)} dB. Output RMS: ${Number(job.output_rms_db || 0).toFixed(1)} dB.</p>
  `;

  const card = document.createElement("div");
  card.className = "result-row";
  card.innerHTML = `
    <div class="result-name">${job.download_name || "Best optimized vocal"}</div>
    <div class="result-meta">${job.detected_only ? "This is the original vocal plus weak-word detection, because regeneration was unavailable for this run." : "This is the best detected-and-regenerated vocal built from local weak-word replacements."}</div>
    ${buildManagedAudioMarkup(job.result_url, job.download_name || "Best optimized vocal", "Touch-up result")}
    <div class="result-links">
      <a class="result-link" href="${job.result_url}" download="${job.download_name || "touched-up-word.wav"}">Download file</a>
    </div>
  `;
  touchUpResults.appendChild(card);
  hydrateManagedAudio(touchUpResults);
}

async function pollJob(jobId) {
  const endpoint =
    state.currentJobKind === "aligned-pth-conversion"
      ? `/api/jobs/${jobId}`
      : `/api/master-conversion/jobs/${jobId}`;
  const response = await fetch(endpoint);
  const job = await response.json();
  if (state.currentJobKind === "aligned-pth-conversion") {
    const totalFiles = Math.max(1, Number(job.total_files || 0));
    const completedFiles = Number(job.completed_files || 0);
    const percent = Math.round((completedFiles / totalFiles) * 100);
    const renderableResults = Array.isArray(job.results) ? [...job.results] : [];
    if (!renderableResults.length && job.result_url) {
      renderableResults.push({
        name: job.current_file || job.download_name || "converted.wav",
        url: job.result_url,
        download_name: job.download_name || "converted.wav",
        sample_rate: job.sample_rate || 0,
        timings: job.timings || {},
      });
    }
    if (job.status === "queued") {
      setStatus("running", "Queued", job.message, 8);
    } else if (job.status === "running") {
      setStatus("running", "Paired conversion running", job.message, Math.max(percent, 12));
    } else if (job.status === "completed") {
      setStatus("completed", "Paired conversion complete", job.message, 100);
      if (!renderableResults.length) {
        if ((state.convertResultRetryCount || 0) < 1) {
          state.convertResultRetryCount += 1;
          resultSummary.classList.add("hidden");
          resultSummary.innerHTML = "";
          results.innerHTML = "";
          clearInterval(state.pollHandle);
          state.pollHandle = setTimeout(() => pollJob(state.currentJobId), 1200);
          return;
        }
        resultSummary.classList.remove("hidden");
        resultSummary.innerHTML = `
          <strong>${state.currentConversionModelSystem || "Direct conversion"} finished</strong>
          <p>The converted file completed, but the output list did not return a downloadable audio URL.</p>
        `;
        results.innerHTML = "";
        clearInterval(state.pollHandle);
        state.pollHandle = null;
        return;
      }
      state.convertResultRetryCount = 0;
      try {
        renderDirectConversionCompletion(job, renderableResults);
      } catch (error) {
        renderDirectConversionFallback(job, renderableResults, error);
      }
      clearInterval(state.pollHandle);
      state.pollHandle = null;
    } else if (job.status === "failed") {
      setStatus("failed", "Failed", job.error || job.message, percent);
      clearInterval(state.pollHandle);
      state.pollHandle = null;
    }
    return;
  }

  const percent = Number(job.progress || 0);

  if (job.status === "queued") {
    setStatus("running", "Queued", job.message, 8);
  } else if (job.status === "running") {
    const stage = job.stage ? `${job.stage}: ` : "";
    setStatus("running", "Master Conversion running", `${stage}${job.message}`, Math.max(percent, 12));
  } else if (job.status === "completed") {
    setStatus("completed", "Master Conversion complete", job.message, 100);
    renderResults(job);
    clearInterval(state.pollHandle);
    state.pollHandle = null;
  } else if (job.status === "failed") {
    setStatus("failed", "Failed", job.error || job.message, percent);
    renderResults(job);
    clearInterval(state.pollHandle);
    state.pollHandle = null;
  }
}

async function pollGenerateJob(jobId) {
  const response = await fetch(`/api/generate/jobs/${jobId}`);
  const job = await response.json();
  renderGenerateResult(job);

  if (job.status === "queued") {
    setGenerateStatus("running", "Queued", job.message, 8);
  } else if (job.status === "running") {
    setGenerateStatus(
      "running",
      "Repairing pronunciation...",
      job.message,
      Math.max(Number(job.progress || 0), 12),
    );
  } else if (job.status === "completed") {
    setGenerateStatus("completed", "Done", job.message, 100);
    clearInterval(state.generatePollHandle);
    state.generatePollHandle = null;
  } else if (job.status === "failed") {
    setGenerateStatus("failed", "Failed", job.error || job.message, Number(job.progress || 0));
    clearInterval(state.generatePollHandle);
    state.generatePollHandle = null;
  }
}

async function pollIsolatorJob(jobId) {
  const response = await fetch(`/api/isolator/jobs/${jobId}`);
  const job = await response.json();
  renderIsolatorResult(job);
  const currentFileSummary = job.current_file
    ? ` Current file: ${job.current_file}.`
    : "";
  const sourceFiles = Array.isArray(job.source_files) ? job.source_files : [];
  const sourceSummary = sourceFiles.length
    ? ` Queue: ${sourceFiles.join(", ")}.`
    : "";

  if (job.status === "queued") {
    setIsolatorStatus("running", "Queued", `${job.message}${currentFileSummary}${sourceSummary}`, 8);
  } else if (job.status === "running") {
    setIsolatorStatus(
      "running",
      "Isolating vocals...",
      `${job.message}${currentFileSummary}${sourceSummary}`,
      Math.max(Number(job.progress || 0), 12),
    );
  } else if (job.status === "completed") {
    setIsolatorStatus("completed", "Done", job.message, 100);
    clearInterval(state.isolatorPollHandle);
    state.isolatorPollHandle = null;
  } else if (job.status === "failed") {
    setIsolatorStatus("failed", "Failed", job.error || job.message, Number(job.progress || 0));
    clearInterval(state.isolatorPollHandle);
    state.isolatorPollHandle = null;
  }
}

async function pollMasteringJob(jobId) {
  const response = await fetch(`/api/mastering/jobs/${jobId}`);
  const job = await response.json();
  renderMasteringResult(job);

  if (job.status === "queued") {
    setMasteringStatus("running", "Queued", job.message, 8);
  } else if (job.status === "running") {
    setMasteringStatus(
      "running",
      "Matching EQ...",
      job.message,
      Math.max(Number(job.progress || 0), 12),
    );
  } else if (job.status === "completed") {
    setMasteringStatus("completed", "Done", job.message, 100);
    clearInterval(state.masteringPollHandle);
    state.masteringPollHandle = null;
  } else if (job.status === "failed") {
    setMasteringStatus("failed", "Failed", job.error || job.message, Number(job.progress || 0));
    clearInterval(state.masteringPollHandle);
    state.masteringPollHandle = null;
  }
}

async function pollOptimizeJob(jobId) {
  const response = await fetch(`/api/optimize/jobs/${jobId}`);
  const job = await response.json();
  renderOptimizeResult(job);
  const currentSummary = job.current_file ? ` Current file: ${job.current_file}.` : "";
  const progress = Math.max(Number(job.progress || 0), 12);

  if (job.status === "queued") {
    setOptimizeStatus("running", "Queued", `${job.message}${currentSummary}`, 8);
  } else if (job.status === "running") {
    setOptimizeStatus("running", "Scoring and stitching...", `${job.message}${currentSummary}`, progress);
  } else if (job.status === "completed") {
    setOptimizeStatus("completed", "Done", job.message, 100);
    clearInterval(state.optimizePollHandle);
    state.optimizePollHandle = null;
  } else if (job.status === "failed") {
    setOptimizeStatus("failed", "Failed", job.error || job.message, Number(job.progress || 0));
    clearInterval(state.optimizePollHandle);
    state.optimizePollHandle = null;
  }
}

async function pollApiComposeJob(jobId) {
  const response = await fetch(`/api/api-compose/jobs/${jobId}`);
  const job = await response.json();
  renderApiComposeResult(job);

  if (job.status === "queued") {
    setApiComposeStatus("running", "Queued", job.message, 8);
  } else if (job.status === "running") {
    setApiComposeStatus(
      "running",
      "Sending request...",
      job.message,
      Math.max(Number(job.progress || 0), 12),
    );
  } else if (job.status === "completed") {
    setApiComposeStatus("completed", "Done", job.message, 100);
    clearInterval(state.apiComposePollHandle);
    state.apiComposePollHandle = null;
  } else if (job.status === "failed") {
    setApiComposeStatus("failed", "Failed", job.error || job.message, Number(job.progress || 0));
    clearInterval(state.apiComposePollHandle);
    state.apiComposePollHandle = null;
  }
}

async function pollTouchUpJob(jobId) {
  const response = await fetch(`/api/touchup/jobs/${jobId}`);
  const job = await response.json();
  renderTouchUpResult(job);
  const scoreSummary = Number.isFinite(Number(job.best_similarity_score))
    ? ` Best ordered-word similarity: ${Number(job.best_similarity_score || 0).toFixed(2)}%.`
    : "";
  const testedSummary = Number(job.repair_attempts || job.variants_tested || 0)
    ? ` Repair attempts: ${Number(job.repair_attempts || job.variants_tested || 0)}.`
    : "";
  const wordReportSummary = job.best_word_report
    ? ` ${job.best_word_report}`
    : "";
  const letterReportSummary = job.best_letter_report
    ? ` ${job.best_letter_report}`
    : "";
  const regenerationSummary = job.regeneration_reason
    ? ` ${job.regeneration_reason}`
    : "";

  if (job.status === "queued") {
    setTouchUpStatus("running", "Queued", `${job.message}${scoreSummary}${testedSummary}`, 8);
  } else if (job.status === "running") {
    setTouchUpStatus(
      "running",
      job.stop_requested
        ? "Stopping..."
        : (job.mode === "smart-removal"
          ? "Removing non-lyric noise..."
          : "Detecting and repairing weak words..."),
      `${job.message}${scoreSummary}${testedSummary}${wordReportSummary}${letterReportSummary}${regenerationSummary}`,
      Math.max(Number(job.progress || 0), 12),
    );
  } else if (job.status === "completed") {
    setTouchUpStatus("completed", "Done", `${job.message}${scoreSummary}${testedSummary}${wordReportSummary}${letterReportSummary}${regenerationSummary}`, 100);
    stopTouchUpButton.disabled = true;
    clearInterval(state.touchUpPollHandle);
    state.touchUpPollHandle = null;
  } else if (job.status === "failed") {
    setTouchUpStatus("failed", "Failed", job.error || job.message, Number(job.progress || 0));
    stopTouchUpButton.disabled = true;
    clearInterval(state.touchUpPollHandle);
    state.touchUpPollHandle = null;
  }
}

async function pollTrainingJob(jobId) {
  if (!jobId || state.trainingPollInFlight) {
    return;
  }
  state.trainingPollInFlight = true;
  try {
    const response = await fetch(`/api/training/jobs/${jobId}`);
    const job = await response.json();
    const percent = Number(job.progress || 0);
    const historyText = Array.isArray(job.log_history) && job.log_history.length
      ? job.log_history.join("\n")
      : (job.log_tail || job.message || "Training logs will appear here.");
    if (trainingLog.textContent !== historyText) {
      trainingLog.textContent = historyText;
      trainingLog.scrollTop = trainingLog.scrollHeight;
    }
    renderTrainingCheckpointPreview(job);

    if (job.status === "queued") {
      setTrainingStatus("running", "Queued", job.message, 8);
      stopTrainingButton.disabled = false;
    } else if (job.status === "running") {
      const stageLabel = job.stage ? `${job.stage}: ` : "";
      setTrainingStatus(
        "running",
        job.stop_requested ? "Stopping..." : "Building PIPA",
        `${stageLabel}${job.message}`,
        Math.max(percent, 10),
      );
      stopTrainingButton.disabled = Boolean(job.stop_requested);
    } else if (job.status === "completed") {
      setTrainingStatus("completed", "PIPA complete", job.message, 100);
      renderTrainingResult(job);
      await loadModels();
      stopTrainingButton.disabled = true;
      state.currentTrainingJobId = null;
      clearInterval(state.trainingPollHandle);
      state.trainingPollHandle = null;
    } else if (job.status === "stopped") {
      setTrainingStatus("completed", "Training stopped", job.message, percent);
      trainingLog.textContent = historyText;
      renderTrainingResult(job);
      stopTrainingButton.disabled = true;
      state.currentTrainingJobId = null;
      clearInterval(state.trainingPollHandle);
      state.trainingPollHandle = null;
    } else if (job.status === "failed") {
      setTrainingStatus("failed", "PIPA build failed", job.error || job.message, percent);
      trainingLog.textContent = job.error || historyText || "Training failed.";
      renderTrainingResult(job);
      stopTrainingButton.disabled = true;
      state.currentTrainingJobId = null;
      clearInterval(state.trainingPollHandle);
      state.trainingPollHandle = null;
    }
  } finally {
    state.trainingPollInFlight = false;
  }
}

async function pollDetagJob(jobId) {
  const response = await fetch(`/api/detag/jobs/${jobId}`);
  const job = await response.json();
  renderDetagResult(job);

  if (job.status === "queued") {
    setDetagStatus("running", "Queued", job.message, 8);
  } else if (job.status === "running") {
    setDetagStatus(
      "running",
      "Detagging...",
      job.message,
      Math.max(Number(job.progress || 0), 12),
    );
  } else if (job.status === "completed") {
    setDetagStatus("completed", "Done", job.message, 100);
    clearInterval(state.detagPollHandle);
    state.detagPollHandle = null;
  } else if (job.status === "failed") {
    setDetagStatus("failed", "Failed", job.error || job.message, Number(job.progress || 0));
    clearInterval(state.detagPollHandle);
    state.detagPollHandle = null;
  }
}

async function startConversion() {
  const leadBuilderModels = getLeadBuilderModels();
  if (!leadBuilderModels.length) {
    setStatus(
      "failed",
      "No conversion voices found",
      "Add a classic RVC .pth voice or train a base-first or paired conversion voice in Voice Builder before running Lead Builder.",
    );
    return;
  }
  if (!state.files.length) {
    setStatus(
      "failed",
      "No SUNO vocal selected",
      "Attach one isolated SUNO vocal file first.",
    );
    return;
  }
  const selectedModel = getLeadBuilderSelectedModel();
  if (!selectedModel || !isLeadBuilderConversionModel(selectedModel)) {
    setStatus(
      "failed",
      "Wrong voice type selected",
      "Lead Builder only runs classic RVC .pth voices plus the base-first and paired conversion voices built in Voice Builder.",
      0,
    );
    return;
  }
  const selectedModelLabel = selectedModel.label || selectedModel.name || "selected voice";
  const selectedModelSystem = selectedModel.system || "Direct conversion";
  state.currentConversionModelName = selectedModel.name || "";
  state.currentConversionModelLabel = selectedModelLabel;
  state.currentConversionModelSystem = selectedModelSystem;

  const data = new FormData();
  data.append("model_name", modelSelect.value);
  const sourceFile = state.files[0];
  data.append("output_mode", "single");
  data.append("quality_preset", qualityPreset.value);
  data.append("output_format", "wav");
  data.append("preprocess_mode", "off");
  data.append("preprocess_strength", "10");
  data.append("speaker_id", "0");
  data.append("transpose", "0");
  data.append("pitch_method", "");
  data.append("index_path", "");
  data.append("index_rate", "-1");
  data.append("filter_radius", "-1");
  data.append("resample_sr", "0");
  data.append("rms_mix_rate", "-1");
  data.append("protect", "-1");
  data.append("crepe_hop_length", "-1");
  data.append("files", sourceFile, sourceFile.name);

  startButton.disabled = true;
  state.convertResultRetryCount = 0;
  setStatus(
    "running",
    "Uploading SUNO vocal",
    `Sending the isolated SUNO vocal and ${selectedModelLabel} to the backend for direct conversion...`,
    6,
  );
  results.innerHTML = "";
  resultSummary.classList.add("hidden");
  previewMeta.classList.add("hidden");
  previewMeta.innerHTML = "";
  previewPlayer.pause();
  previewPlayer.removeAttribute("src");
  previewPlayer.load();
  previewPlayer.classList.add("hidden");
  setManagedAudioVisible(previewPlayer, false);
  setPreviewStatus(
    "running",
    "Waiting for converted output",
    "The finished direct conversion will appear here the moment the backend returns the file.",
    6,
  );

  try {
    state.currentJobKind = "aligned-pth-conversion";
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not start the job.");
    }
    state.currentJobId = payload.job_id;
    await pollJob(state.currentJobId);
    state.pollHandle = setInterval(() => pollJob(state.currentJobId), 1500);
  } catch (error) {
    setStatus(
      "failed",
      "Could not start direct conversion",
      error.message || String(error),
      0,
    );
  } finally {
    startButton.disabled = false;
  }
}

async function startGenerate() {
  if (!getPersonaRepairModels().length) {
    setGenerateStatus(
      "failed",
      "No persona voices found",
      "Train or add a Persona v1.0 voice before repairing.",
      0,
    );
    return;
  }
  if (!state.generateGuideFile) {
    setGenerateStatus(
      "failed",
      "No reference vocal selected",
      "Drop one clean reference vocal file into the upload area first.",
      0,
    );
    return;
  }
  if (!(generateLyrics.value || "").trim()) {
    setGenerateStatus(
      "failed",
      "Missing lyrics",
      "Paste the intended lyrics so the repair system can score the words correctly.",
      0,
    );
    return;
  }

  const data = new FormData();
  data.append("model_name", generateModelSelect.value);
  data.append("lyrics", generateLyrics.value);
  data.append("guide_key", generateGuideKey.value || "");
  data.append("target_key", generateTargetKey.value || "");
  data.append("guide_bpm", generateGuideBpm.value || "0");
  data.append("target_bpm", generateTargetBpm.value || "0");
  data.append("quality_preset", generateQualityPreset.value || "balanced");
  data.append("preprocess_mode", generatePreprocessMode.value || "off");
  data.append("preprocess_strength", generatePreprocessStrength.value || "9");
  data.append("guide_file", state.generateGuideFile, state.generateGuideFile.name);

  startGenerateButton.disabled = true;
  generateResultSummary.classList.add("hidden");
  generateResults.innerHTML = "";
  setGenerateStatus(
    "running",
    "Uploading reference vocal",
    "Sending the vocal, lyrics, and selected voice to the backend...",
    6,
  );

  try {
    const response = await fetch("/api/generate/jobs", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not start pronunciation repair.");
    }
    state.currentGenerateJobId = payload.job_id;
    await pollGenerateJob(state.currentGenerateJobId);
    state.generatePollHandle = setInterval(
      () => pollGenerateJob(state.currentGenerateJobId),
      1500,
    );
  } catch (error) {
    setGenerateStatus(
      "failed",
      "Could not start pronunciation repair",
      error.message || String(error),
      0,
    );
  } finally {
    startGenerateButton.disabled = false;
  }
}

async function startTraining() {
  const selectedOutputMode = trainingOutputMode.value || "classic-rvc-support";
  const modeConfig = getTrainingModeConfig(selectedOutputMode);
  const trainingSummary = summarizeTrainingFiles(state.trainingFiles, selectedOutputMode);
  const directPairMode = selectedOutputMode === "persona-aligned-pth" || selectedOutputMode === "concert-remaster-paired";
  const classicSupportMode = selectedOutputMode === "classic-rvc-support";
  if (!state.trainingFiles.length) {
    setTrainingStatus(
      "failed",
      "No clips selected",
      "Drop training clips into the upload area first.",
      0,
    );
    return;
  }
  if (selectedOutputMode === "persona-aligned-pth" && trainingSummary.baseCount < 1) {
    setTrainingStatus(
      "failed",
      "Missing BASE clips",
      "Add at least one long target identity clip named with the BASE prefix before starting training.",
      0,
    );
    return;
  }
  if (classicSupportMode && trainingSummary.baseCount < 1) {
    setTrainingStatus(
      "failed",
      "Missing BASE clips",
      "Add at least one clean target identity clip named with the BASE prefix before starting the classic RVC run.",
      0,
    );
    return;
  }
  if (directPairMode && (trainingSummary.targetCount < 1 || trainingSummary.sourceCount < 1 || trainingSummary.matchedPairCount < 1)) {
    setTrainingStatus(
      "failed",
      `No matched ${modeConfig.targetName}/${modeConfig.sourceName} pairs`,
      `Add matching aligned files like ${modeConfig.targetName}_phrase01.wav and ${modeConfig.sourceName}_phrase01.wav before building the model.`,
      0,
    );
    return;
  }

  const data = new FormData();
  data.append("experiment_name", trainingName.value || "voice-model");
  data.append("sample_rate", trainingSampleRate.value);
  data.append("version", trainingVersion.value);
  data.append("f0_method", trainingF0Method.value);
  data.append("output_mode", selectedOutputMode);
  data.append("epoch_mode", "fixed");
  data.append("total_epochs", String(clampTrainingEpochInput(trainingEpochs, 600)));
  data.append("save_every_epoch", trainingSaveEvery.value || "25");
  data.append("batch_size", trainingBatchSize.value || "4");
  data.append("crepe_hop_length", trainingCrepeHopLength.value || "128");
  data.append("resume_selection_name", trainingResumePackageSelect?.value || "");
  data.append("start_phase", trainingStartPhase?.value || "auto");
  for (const file of state.trainingFiles) {
    data.append("files", file, file.name);
  }
  for (const file of state.trainingPlanFiles) {
    data.append("plan_files", file, file.name);
  }
  for (const file of state.trainingTranscriptFiles) {
    data.append("transcript_files", file, file.name);
  }

  startTrainingButton.disabled = true;
  stopTrainingButton.disabled = true;
  setTrainingStatus(
    "running",
    "Uploading dataset",
    classicSupportMode
      ? "Sending your BASE truth clips and SUNO checkpoint-audition clips to the backend..."
      : directPairMode
      ? (selectedOutputMode === "concert-remaster-paired"
        ? "Sending your CONCERT/CD aligned audio clips to the backend..."
        : "Sending your BASE/TARGET/SUNO aligned audio clips to the backend...")
      : "Sending your audio, persona plan, and transcript data to the backend...",
    6,
  );
  trainingResultSummary.classList.add("hidden");
  trainingLog.textContent = "Training logs will appear here.";
  renderTrainingCheckpointPreview({});

  try {
    const response = await fetch("/api/training/jobs", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not start training.");
    }
    trainingName.value = payload.experiment_name || trainingName.value;
    state.currentTrainingJobId = payload.job_id;
    state.trainingPollInFlight = false;
    stopTrainingButton.disabled = false;
    await pollTrainingJob(state.currentTrainingJobId);
    state.trainingPollHandle = setInterval(
      () => pollTrainingJob(state.currentTrainingJobId),
      500,
    );
  } catch (error) {
    setTrainingStatus(
      "failed",
      "Could not start training",
      error.message || String(error),
      0,
    );
    stopTrainingButton.disabled = true;
  } finally {
    startTrainingButton.disabled = false;
  }
}

async function stopTraining() {
  if (!state.currentTrainingJobId) {
    return;
  }

  stopTrainingButton.disabled = true;
  try {
    const response = await fetch(`/api/training/jobs/${state.currentTrainingJobId}/stop`, {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not stop training.");
    }
    const stageLabel = payload.stage ? `${payload.stage}: ` : "";
    setTrainingStatus(
      "running",
      "Stop requested",
      `${stageLabel}${payload.message || "Finishing the current chunk..."}`,
      Math.max(Number(payload.progress || 0), 10),
    );
  } catch (error) {
    stopTrainingButton.disabled = false;
    setTrainingStatus(
      "failed",
      "Could not stop training",
      error.message || String(error),
      0,
    );
  }
}

async function startDetag() {
  if (!state.detagVoices.length) {
    setDetagStatus(
      "failed",
      "No voices found",
      "You need a voice reference folder before detag can run.",
      0,
    );
    return;
  }
  const selectedVoice = state.detagVoices.find((voice) => voice.id === detagVoiceSelect.value);
  if (!selectedVoice) {
    setDetagStatus(
      "failed",
      "No usable voice selected",
      "Choose a voice that has reference clips before running detag.",
      0,
    );
    return;
  }
  if (!selectedVoice.ready) {
    setDetagStatus(
      "failed",
      "Voice needs reference clips",
      "This model appears in weights, but detag still needs matching clips in logs/<voice>/0_gt_wavs.",
      0,
    );
    return;
  }
  if (!state.detagFile) {
    setDetagStatus(
      "failed",
      "No file selected",
      "Drop one audio file into the upload area first.",
      0,
    );
    return;
  }

  const data = new FormData();
  data.append("voice_id", selectedVoice.id);
  data.append("strength", detagStrength.value);
  data.append("file", state.detagFile, state.detagFile.name);

  startDetagButton.disabled = true;
  detagResultSummary.classList.add("hidden");
  detagResults.innerHTML = "";
  setDetagStatus("running", "Uploading file", "Sending your audio to the backend...", 6);

  try {
    const response = await fetch("/api/detag/jobs", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not start detag.");
    }
    state.currentDetagJobId = payload.job_id;
    await pollDetagJob(state.currentDetagJobId);
    state.detagPollHandle = setInterval(
      () => pollDetagJob(state.currentDetagJobId),
      1500,
    );
  } catch (error) {
    setDetagStatus(
      "failed",
      "Could not start detag",
      error.message || String(error),
      0,
    );
  } finally {
    startDetagButton.disabled = false;
  }
}

async function startIsolator() {
  if (!state.isolatorFiles.length) {
    setIsolatorStatus(
      "failed",
      "No files selected",
      "Drop one or more files into the upload area first.",
      0,
    );
    return;
  }

  const data = new FormData();
  data.append("mode", isolatorMode.value);
  data.append("input_type", isolatorInputType.value);
  data.append("strength", isolatorStrength.value);
  data.append("deecho", String(isolatorDeecho.value === "true"));
  data.append("width_focus", String(isolatorWidthFocus.value === "true"));
  data.append("clarity_preserve", isolatorClarityPreserve.value);
  for (const file of state.isolatorFiles) {
    data.append("files", file, file.name);
  }

  startIsolatorButton.disabled = true;
  isolatorResultSummary.classList.add("hidden");
  isolatorResults.innerHTML = "";
  const queuedSummary = state.isolatorFiles.map((file) => file.name).join(", ");
  setIsolatorStatus(
    "running",
    "Uploading files",
    `Sending ${state.isolatorFiles.length} file${state.isolatorFiles.length === 1 ? "" : "s"} to the backend. Queue: ${queuedSummary}.`,
    6,
  );

  try {
    const response = await fetch("/api/isolator/jobs", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not start the isolator.");
    }
    state.currentIsolatorJobId = payload.job_id;
    await pollIsolatorJob(state.currentIsolatorJobId);
    state.isolatorPollHandle = setInterval(
      () => pollIsolatorJob(state.currentIsolatorJobId),
      1500,
    );
  } catch (error) {
    setIsolatorStatus(
      "failed",
      "Could not start isolation",
      error.message || String(error),
      0,
    );
  } finally {
    startIsolatorButton.disabled = false;
  }
}

async function startMastering() {
  if (!state.masteringSourceFile) {
    setMasteringStatus(
      "failed",
      "No source file selected",
      "Drop the file you want to master first.",
      0,
    );
    return;
  }
  if (!state.masteringReferenceFiles.length) {
    setMasteringStatus(
      "failed",
      "No reference files selected",
      "Drop one or more mastered reference files first.",
      0,
    );
    return;
  }

  const data = new FormData();
  data.append("resolution", masteringResolution.value);
  data.append("source_file", state.masteringSourceFile, state.masteringSourceFile.name);
  for (const file of state.masteringReferenceFiles) {
    data.append("reference_files", file, file.name);
  }

  startMasteringButton.disabled = true;
  masteringResultSummary.classList.add("hidden");
  masteringResults.innerHTML = "";
  masteringProfileCard.classList.add("hidden");
  masteringProfileChart.innerHTML = "";
  masteringProfileMeta.textContent = "";
  setMasteringStatus(
    "running",
    "Uploading files",
    `Sending the source track and ${state.masteringReferenceFiles.length} reference master${state.masteringReferenceFiles.length === 1 ? "" : "s"} to the backend...`,
    6,
  );

  try {
    const response = await fetch("/api/mastering/jobs", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not start mastering.");
    }
    state.currentMasteringJobId = payload.job_id;
    await pollMasteringJob(state.currentMasteringJobId);
    state.masteringPollHandle = setInterval(
      () => pollMasteringJob(state.currentMasteringJobId),
      1500,
    );
  } catch (error) {
    setMasteringStatus(
      "failed",
      "Could not start mastering",
      error.message || String(error),
      0,
    );
  } finally {
    startMasteringButton.disabled = false;
  }
}

async function startOptimize() {
  if (!state.optimizeFiles.length) {
    setOptimizeStatus(
      "failed",
      "No files selected",
      "Drop one or more vocal files into the upload area first.",
      0,
    );
    return;
  }
  if (!(optimizeLyrics.value || "").trim()) {
    setOptimizeStatus(
      "failed",
      "Lyrics missing",
      "Paste the intended song lyrics before stitching.",
      0,
    );
    return;
  }

  const data = new FormData();
  data.append("max_cut_db", optimizeStrength.value);
  data.append("lyrics", optimizeLyrics.value.trim());
  for (const file of state.optimizeFiles) {
    data.append("files", file, file.name);
  }

  startOptimizeButton.disabled = true;
  optimizeResultSummary.classList.add("hidden");
  optimizeResults.innerHTML = "";
  setOptimizeStatus(
    "running",
    "Uploading files",
    `Sending ${state.optimizeFiles.length} vocal take${state.optimizeFiles.length === 1 ? "" : "s"} for lyric-aware stitching...`,
    6,
  );

  try {
    const response = await fetch("/api/optimize/jobs", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not start stitching.");
    }
    state.currentOptimizeJobId = payload.job_id;
    await pollOptimizeJob(state.currentOptimizeJobId);
    state.optimizePollHandle = setInterval(
      () => pollOptimizeJob(state.currentOptimizeJobId),
      1500,
    );
  } catch (error) {
    setOptimizeStatus(
      "failed",
      "Could not start stitch job",
      error.message || String(error),
      0,
    );
  } finally {
    startOptimizeButton.disabled = false;
  }
}

async function createAlbumProject() {
  const name = (albumName?.value || "").trim();

  createAlbumButton.disabled = true;
  setAlbumStatus("running", "Creating project", "Creating album workspace and version log...", 15);
  try {
    const data = new FormData();
    data.append("name", name || "Album Project");
    const response = await fetch("/api/albums/projects", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not create album project.");
    }

    state.currentAlbumProjectId = payload.project?.id || null;
    await loadAlbumProjects(false);
    renderAlbumProject(payload.project || null);
    renderAlbumProjectSelector();
    setAlbumStatus("completed", "Project created", "Album project ready. Upload tracks to start versioned mapping.", 100);
  } catch (error) {
    setAlbumStatus("failed", "Could not create project", error.message || String(error), 0);
  } finally {
    createAlbumButton.disabled = false;
  }
}

async function uploadAlbumSongVersion(songIndex, file) {
  if (!state.currentAlbumProjectId) {
    setAlbumStatus("failed", "No project selected", "Create or select a project first.", 0);
    return;
  }
  const data = new FormData();
  data.append("file", file, file.name);
  setAlbumStatus("running", `Uploading Track ${songIndex}`, "Saving version and rebuilding album preview...", 35);
  try {
    const response = await fetch(
      `/api/albums/projects/${state.currentAlbumProjectId}/songs/${songIndex}/versions`,
      {
        method: "POST",
        body: data,
      },
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not upload song version.");
    }

    renderAlbumProject(payload.project || null);
    await loadAlbumProjects(false);
    renderAlbumProjectSelector();
    setAlbumStatus(
      "completed",
      `Track ${songIndex} updated`,
      `Saved as V${Number(payload.version || 0)}. Older versions remain available from the version dropdown.`,
      100,
    );
  } catch (error) {
    setAlbumStatus("failed", `Track ${songIndex} failed`, error.message || String(error), 0);
  }
}

async function bulkMapAlbumSongs() {
  if (!state.currentAlbumProjectId) {
    setAlbumStatus("failed", "No project selected", "Create or select a project first.", 0);
    return;
  }
  if (!state.albumBulkFiles.length) {
    setAlbumStatus("failed", "No files selected", "Drop track files first, then upload.", 0);
    return;
  }

  albumBulkUploadButton.disabled = true;
  setAlbumStatus("running", "Uploading tracks", "Appending files and rebuilding the album preview...", 20);
  try {
    const data = new FormData();
    for (const file of state.albumBulkFiles) {
      data.append("files", file, file.name);
    }
    const response = await fetch(`/api/albums/projects/${state.currentAlbumProjectId}/songs/bulk`, {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not upload tracks.");
    }

    renderAlbumProject(payload.project || null);
    await loadAlbumProjects(false);
    renderAlbumProjectSelector();
    state.albumBulkFiles = [];
    renderAlbumBulkFiles();
    setAlbumStatus(
      "completed",
      "Upload complete",
      `Added ${Number(payload.uploaded_count || 0)} new track(s). Ignored ${Number(payload.ignored_count || 0)}.`,
      100,
    );
  } catch (error) {
    setAlbumStatus("failed", "Upload failed", error.message || String(error), 0);
  } finally {
    albumBulkUploadButton.disabled = false;
  }
}

async function checkApiComposeHealth() {
  const baseUrl = (apiComposeEndpointUrl.value || "").trim();
  if (!baseUrl) {
    setApiComposeStatus("failed", "Endpoint missing", "Enter the ACE-Step API URL first.", 0);
    return;
  }
  setApiComposeStatus("running", "Checking server", "Pinging local ACE-Step API health endpoint...", 15);
  try {
    const response = await fetch(`/api/api-compose/provider-health?endpoint_url=${encodeURIComponent(baseUrl)}`);
    const payload = await response.json();
    if (payload.ok) {
      setApiComposeStatus("completed", "Server reachable", `ACE-Step API is online at ${payload.url}`, 100);
    } else {
      setApiComposeStatus("failed", "Server unavailable", payload.body_preview || "Could not reach ACE-Step API.", 0);
    }
  } catch (error) {
    setApiComposeStatus("failed", "Health check failed", error.message || String(error), 0);
  }
}

async function startApiCompose() {
  if (!state.apiComposeMidiFile) {
    setApiComposeStatus("failed", "No MIDI file selected", "Drop one MIDI file first.", 0);
    return;
  }
  if (!(apiComposeLyrics.value || "").trim()) {
    setApiComposeStatus("failed", "Lyrics missing", "Paste lyrics before sending.", 0);
    return;
  }
  if (!(apiComposeEndpointUrl.value || "").trim()) {
    setApiComposeStatus("failed", "Endpoint missing", "Enter your ACE-Step API endpoint URL.", 0);
    return;
  }

  const data = new FormData();
  data.append("endpoint_url", apiComposeEndpointUrl.value.trim());
  data.append("lyrics", apiComposeLyrics.value.trim());
  data.append("auth_header", (apiComposeAuthHeader.value || "").trim());
  data.append("api_key", (apiComposeApiKey.value || "").trim());
  data.append("extra_json", (apiComposeExtraJson.value || "").trim());
  data.append("midi_file", state.apiComposeMidiFile, state.apiComposeMidiFile.name);
  if (state.apiComposeBeatFile) {
    data.append("beat_file", state.apiComposeBeatFile, state.apiComposeBeatFile.name);
  }

  startApiComposeButton.disabled = true;
  apiComposeResultSummary.classList.add("hidden");
  apiComposeResultSummary.innerHTML = "";
  apiComposeResults.innerHTML = "";
  apiComposeResponse.textContent = "ACE-Step response will appear here.";
  setApiComposeStatus("running", "Submitting task", "Sending payload to local ACE-Step workflow...", 6);

  try {
    const response = await fetch("/api/api-compose/jobs", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not start ACE-Step job.");
    }
    state.currentApiComposeJobId = payload.job_id;
    await pollApiComposeJob(state.currentApiComposeJobId);
    state.apiComposePollHandle = setInterval(
      () => pollApiComposeJob(state.currentApiComposeJobId),
      1500,
    );
  } catch (error) {
    setApiComposeStatus(
      "failed",
      "Could not start ACE-Step",
      error.message || String(error),
      0,
    );
  } finally {
    startApiComposeButton.disabled = false;
  }
}

async function startTouchUp() {
  const sourceWord = (touchUpSourceWord.value || "").trim();
  const selectedMode = touchUpMode?.value || "smart-removal";

  if (!sourceWord) {
    setTouchUpStatus("failed", "Missing target lyrics", "Paste the exact lyrics for the vocal first.", 0);
    return;
  }
  if (!state.touchUpSourceFile) {
    setTouchUpStatus("failed", "No vocal file selected", "Drop the vocal file first.", 0);
    return;
  }

  const data = new FormData();
  data.append("source_word", sourceWord);
  data.append("mode", selectedMode);
  data.append("strength", touchUpStrength.value);
  data.append("max_target_words", touchUpMaxWords.value || "5");
  data.append("source_file", state.touchUpSourceFile, state.touchUpSourceFile.name);

  startTouchUpButton.disabled = true;
  stopTouchUpButton.disabled = false;
  touchUpResultSummary.classList.add("hidden");
  touchUpResults.innerHTML = "";
    setTouchUpStatus(
      "running",
      "Uploading files",
      selectedMode === "smart-removal"
        ? "Sending the vocal and lyrics to the backend for lyric-aware cleanup..."
        : "Sending the AI vocal and intended lyrics to the backend for weak-word detection...",
      6,
    );

  try {
    const response = await fetch("/api/touchup/jobs", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not start pronunciation optimizer.");
    }
    state.currentTouchUpJobId = payload.job_id;
    await pollTouchUpJob(state.currentTouchUpJobId);
    state.touchUpPollHandle = setInterval(
      () => pollTouchUpJob(state.currentTouchUpJobId),
      1500,
    );
  } catch (error) {
    stopTouchUpButton.disabled = true;
    setTouchUpStatus(
      "failed",
      selectedMode === "smart-removal" ? "Could not start smart removal" : "Could not start pronunciation optimizer",
      error.message || String(error),
      0,
    );
  } finally {
    startTouchUpButton.disabled = false;
  }
}

async function stopTouchUp() {
  if (!state.currentTouchUpJobId) {
    setTouchUpStatus("failed", "No optimizer running", "Start the pronunciation optimizer first.", 0);
    return;
  }

  stopTouchUpButton.disabled = true;
  try {
    const response = await fetch(`/api/touchup/jobs/${state.currentTouchUpJobId}/stop`, {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not stop the optimizer.");
    }
    setTouchUpStatus(
      "running",
      "Stop requested",
      "Finishing the current word region, then the best pronunciation result so far will be returned.",
      Number(touchUpProgressBar.style.width?.replace("%", "") || 0),
    );
  } catch (error) {
    stopTouchUpButton.disabled = false;
    setTouchUpStatus("failed", "Could not stop optimizer", error.message || String(error), 0);
  }
}

function wireDropZone(dropTarget, onFiles) {
  if (!dropTarget) {
    return;
  }
  let dragDepth = 0;

  dropTarget.addEventListener("dragenter", (event) => {
    event.preventDefault();
    event.stopPropagation();
    dragDepth += 1;
    dropTarget.classList.add("dragging");
  });

  dropTarget.addEventListener("dragover", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = "copy";
    }
    dropTarget.classList.add("dragging");
  });

  dropTarget.addEventListener("dragleave", (event) => {
    event.preventDefault();
    event.stopPropagation();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) {
      dropTarget.classList.remove("dragging");
    }
  });

  dropTarget.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    dragDepth = 0;
    dropTarget.classList.remove("dragging");
    onFiles(extractDroppedFiles(event));
  });
}

window.addEventListener("dragover", (event) => {
  event.preventDefault();
});

window.addEventListener("drop", (event) => {
  event.preventDefault();
});

wireDropZone(dropZone, addFiles);
wireDropZone(generateDropZone, setGenerateGuideFile);
wireDropZone(indexDropZone, setIndexFile);
wireDropZone(isolatorDropZone, setIsolatorFile);
wireDropZone(masteringSourceDropZone, setMasteringSourceFile);
wireDropZone(masteringReferenceDropZone, setMasteringReferenceFile);
wireDropZone(optimizeDropZone, setOptimizeFiles);
wireDropZone(albumBulkDropZone, setAlbumBulkFiles);
wireDropZone(apiComposeMidiDropZone, setApiComposeMidiFile);
wireDropZone(apiComposeBeatDropZone, setApiComposeBeatFile);
wireDropZone(touchUpSourceDropZone, setTouchUpSourceFile);
wireDropZone(trainingDropZone, addTrainingFiles);
wireDropZone(trainingPlanDropZone, addTrainingPlanFiles);
wireDropZone(trainingTranscriptDropZone, addTrainingTranscriptFiles);

bindIfPresent(fileInput, "change", (event) => {
  addFiles(event.target.files);
  fileInput.value = "";
});

bindIfPresent(generateGuideFileInput, "change", (event) => {
  setGenerateGuideFile(event.target.files);
  generateGuideFileInput.value = "";
});

bindIfPresent(indexFileInput, "change", (event) => {
  setIndexFile(event.target.files);
  indexFileInput.value = "";
});

bindIfPresent(isolatorFileInput, "change", (event) => {
  setIsolatorFile(event.target.files);
  isolatorFileInput.value = "";
});

bindIfPresent(masteringSourceFileInput, "change", (event) => {
  setMasteringSourceFile(event.target.files);
  masteringSourceFileInput.value = "";
});

bindIfPresent(masteringReferenceFileInput, "change", (event) => {
  setMasteringReferenceFile(event.target.files);
  masteringReferenceFileInput.value = "";
});

bindIfPresent(optimizeFileInput, "change", (event) => {
  setOptimizeFiles(event.target.files);
  optimizeFileInput.value = "";
});

bindIfPresent(albumBulkFileInput, "change", (event) => {
  setAlbumBulkFiles(event.target.files);
  albumBulkFileInput.value = "";
});

bindIfPresent(apiComposeMidiFileInput, "change", (event) => {
  setApiComposeMidiFile(event.target.files);
  apiComposeMidiFileInput.value = "";
});

bindIfPresent(apiComposeBeatFileInput, "change", (event) => {
  setApiComposeBeatFile(event.target.files);
  apiComposeBeatFileInput.value = "";
});

bindIfPresent(touchUpSourceFileInput, "change", (event) => {
  setTouchUpSourceFile(event.target.files);
  touchUpSourceFileInput.value = "";
});

bindIfPresent(trainingFileInput, "change", (event) => {
  addTrainingFiles(event.target.files);
  trainingFileInput.value = "";
});
bindIfPresent(trainingPlanFileInput, "change", (event) => {
  addTrainingPlanFiles(event.target.files);
  trainingPlanFileInput.value = "";
});
bindIfPresent(trainingTranscriptFileInput, "change", (event) => {
  addTrainingTranscriptFiles(event.target.files);
  trainingTranscriptFileInput.value = "";
});
bindIfPresent(trainingOutputMode, "change", syncTrainingRunModeUI);
bindIfPresent(trainingEpochMode, "change", syncTrainingRunModeUI);
bindIfPresent(trainingPackageDownloadSelect, "change", updateTrainingPackageDownloadSummary);
bindIfPresent(downloadTrainingPackageButton, "click", downloadSelectedTrainingPackage);
bindIfPresent(trainingResumePackageSelect, "change", () => {
  updateTrainingResumeSummary();
});
bindIfPresent(trainingEpochs, "input", updateTrainingCurriculumSummary);

bindIfPresent(pitchPreset, "change", () => {
  customPitchField.classList.toggle("hidden", pitchPreset.value !== "custom");
  scheduleAutoPreview();
});

bindIfPresent(qualityPreset, "change", () => {
  applyQualityPresetDefaults();
  updateQualitySummary();
  scheduleAutoPreview();
});
bindIfPresent(masterProfile, "change", updateMasterProfileSummary);
bindIfPresent(outputMode, "change", () => {
  updateOutputModeUI();
  scheduleAutoPreview();
});
bindIfPresent(generateQualityPreset, "change", updateGenerateQualitySummary);
bindIfPresent(isolatorMode, "change", updateIsolatorModeSummary);
bindIfPresent(isolatorInputType, "change", updateIsolatorInputTypeSummary);
bindIfPresent(modelSelect, "change", () => {
  syncModelDefaults();
  scheduleAutoPreview();
});
bindIfPresent(pipaPackageSelect, "change", updatePipaPackageSummary);
bindIfPresent(secondaryModelSelect, "change", scheduleAutoPreview);
bindIfPresent(blendPercentage, "input", () => {
  updateSliderLabels();
  scheduleAutoPreview();
});
bindIfPresent(generateModelSelect, "change", updateGenerateQualitySummary);
bindIfPresent(refreshModelsButton, "click", loadModels);
bindIfPresent(startButton, "click", startConversion);
bindIfPresent(startGenerateButton, "click", startGenerate);
bindIfPresent(startIsolatorButton, "click", startIsolator);
bindIfPresent(startMasteringButton, "click", startMastering);
bindIfPresent(startOptimizeButton, "click", startOptimize);
bindIfPresent(createAlbumButton, "click", createAlbumProject);
bindIfPresent(refreshAlbumsButton, "click", () => loadAlbumProjects(true).catch((error) => {
  setAlbumStatus("failed", "Could not refresh projects", error.message || String(error), 0);
}));
bindIfPresent(albumBulkUploadButton, "click", bulkMapAlbumSongs);
bindIfPresent(checkApiComposeHealthButton, "click", checkApiComposeHealth);
bindIfPresent(startApiComposeButton, "click", startApiCompose);
bindIfPresent(startTouchUpButton, "click", startTouchUp);
bindIfPresent(stopTouchUpButton, "click", stopTouchUp);
bindIfPresent(startTrainingButton, "click", startTraining);
bindIfPresent(stopTrainingButton, "click", stopTraining);
bindIfPresent(showConvertTabButton, "click", () => setActiveTab("convert"));
bindIfPresent(showGenerateTabButton, "click", () => setActiveTab("generate"));
bindIfPresent(showIsolatorTabButton, "click", () => setActiveTab("isolator"));
bindIfPresent(showOptimizeTabButton, "click", () => setActiveTab("optimize"));
bindIfPresent(showAlbumsTabButton, "click", () => setActiveTab("albums"));
bindIfPresent(showApiComposeTabButton, "click", () => setActiveTab("api-compose"));
bindIfPresent(showTouchUpTabButton, "click", () => setActiveTab("touchup"));
bindIfPresent(showMasteringTabButton, "click", () => setActiveTab("mastering"));
bindIfPresent(showTrainingTabButton, "click", () => setActiveTab("training"));
bindIfPresent(railConvertTabButton, "click", () => setActiveTab("convert"));
bindIfPresent(railIsolatorTabButton, "click", () => setActiveTab("isolator"));
bindIfPresent(railMasteringTabButton, "click", () => setActiveTab("mastering"));
bindIfPresent(railTrainingTabButton, "click", () => setActiveTab("training"));
bindIfPresent(albumProjectSelect, "change", () => {
  const nextId = albumProjectSelect.value || "";
  state.currentAlbumProjectId = nextId;
  if (!nextId) {
    renderAlbumProject(null);
    return;
  }
  loadAlbumProject(nextId).catch((error) => {
    setAlbumStatus("failed", "Could not load project", error.message || String(error), 0);
  });
});

document.querySelectorAll("[data-jump-tab]").forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.jumpTab));
});

document.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.repeat) {
    return;
  }
  if (isTypingField(event.target)) {
    return;
  }
  if (!toggleManagedAudioPlayback()) {
    return;
  }
  event.preventDefault();
});

[indexRate, protect, rmsMixRate, filterRadius, preprocessStrength, generatePreprocessStrength, isolatorStrength, isolatorClarityPreserve, masteringResolution, optimizeStrength, touchUpStrength, detagStrength]
  .filter(Boolean)
  .forEach((input) => {
    input.addEventListener("input", () => {
    updateSliderLabels();
    if (
      input === generatePreprocessStrength ||
      input === isolatorStrength ||
      input === isolatorClarityPreserve ||
      input === masteringResolution ||
      input === optimizeStrength ||
      input === touchUpStrength ||
      input === detagStrength
    ) {
      return;
    }
    scheduleAutoPreview();
  });
  });

[outputFormat, pitchMethod].filter(Boolean).forEach((input) => {
  input.addEventListener("change", scheduleAutoPreview);
});

bindIfPresent(preprocessMode, "change", () => {
  updatePreprocessModeSummary();
  scheduleAutoPreview();
});
bindIfPresent(generatePreprocessMode, "change", updateGeneratePreprocessModeSummary);

bindIfPresent(customPitch, "input", scheduleAutoPreview);
bindIfPresent(indexPath, "input", scheduleAutoPreview);

updateSliderLabels();
updateOutputModeUI();
renderFiles();
renderGenerateGuideFile();
renderIndexFile();
renderIsolatorFile();
renderMasteringSourceFile();
renderMasteringReferenceFile();
renderOptimizeFiles();
renderAlbumBulkFiles();
renderAlbumProject(null);
renderApiComposeMidiFile();
renderApiComposeBeatFile();
renderTouchUpSourceFile();
renderTrainingFiles();
renderTrainingPlanFiles();
renderTrainingTranscriptFiles();
hydrateManagedAudio(document);
setManagedAudioMeta(previewPlayer, "Preview", "Auto preview player");
setManagedAudioVisible(previewPlayer, false);
setManagedAudioMeta(trainingCheckpointPreviewPlayer, "Checkpoint audition", "Latest training checkpoint preview");
setManagedAudioVisible(trainingCheckpointPreviewPlayer, false);
setManagedAudioMeta(albumPreviewPlayer, "Album mix", "Current album playback");
setActiveTab("convert");

Promise.all([loadModels(), loadMasterConversionOptions(), loadGenerateOptions(), loadIsolatorOptions(), loadMasteringOptions(), loadOptimizeOptions(), loadAlbumOptions(), loadApiComposeOptions(), loadTrainingOptions()]).then(() => {
  setStatus(
    "idle",
    "Ready for direct conversion",
    "Choose a base-first or paired conversion voice, add one isolated vocal, and run direct conversion when you are ready.",
  );
  setGenerateStatus(
    "idle",
    "Waiting for a reference vocal",
    "Upload one vocal, paste the intended lyrics, choose the voice, and start the repair pass.",
  );
  setPreviewStatus(
    "idle",
    "Preview disabled",
    "Lead Builder now runs only the full direct-conversion path instead of the old 5 second preview.",
  );
  setTouchUpStatus(
    "idle",
    "Waiting for a vocal",
    "Upload a vocal and paste the exact lyrics to keep only lyric-aligned speech.",
  );
  setIsolatorStatus(
    "idle",
    "Waiting for audio",
    "Upload one or more files, then start the isolator.",
  );
  setMasteringStatus(
    "idle",
    "Waiting for files",
    "Upload a source track and one or more mastered references.",
  );
  setOptimizeStatus(
    "idle",
    "Waiting for files",
    "Upload vocal takes + lyrics, then build the stitched best acapella.",
  );
  setAlbumStatus(
    "idle",
    "Waiting for a project",
    "Create an album project, upload tracks, and replace any track later while keeping V0/V1/V2 history.",
  );
  setApiComposeStatus(
    "idle",
    "Waiting for input",
    "Set local endpoint, add MIDI + lyrics, then generate.",
  );
  setTrainingStatus(
    "idle",
    "Waiting for training audio",
    "Upload BASE clips and, if you want fixed auditions, SUNO clips. Then start the build.",
  );
  return loadAlbumProjects(true);
}).catch((error) => {
  setAlbumStatus(
    "failed",
    "Could not load app data",
    error.message || String(error),
    0,
  );
});
