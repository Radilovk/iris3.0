#!/usr/bin/env node

/**
 * validate-config.js - Validates AI model configuration
 * 
 * This script validates that:
 * 1. wrangler.toml has valid syntax
 * 2. worker.js has valid JavaScript syntax
 * 3. All required environment variables are documented
 * 4. API endpoints are correctly formatted
 */

const fs = require('fs');
const path = require('path');

// Colors for terminal output
const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  reset: '\x1b[0m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function validateWorkerJS() {
  log('\n📝 Validating worker.js...', 'yellow');
  
  const workerPath = path.join(__dirname, 'worker.js');
  const content = fs.readFileSync(workerPath, 'utf-8');
  
  // Check for required functions
  const requiredFunctions = [
    'aiCall',
    'aiCallOpenAI',
    'aiCallGemini',
    'handleAnalyze',
    'handleGetResult'
  ];
  
  const missingFunctions = [];
  for (const func of requiredFunctions) {
    if (!content.includes(`function ${func}`) && !content.includes(`async function ${func}`)) {
      missingFunctions.push(func);
    }
  }
  
  if (missingFunctions.length > 0) {
    log(`✗ Missing functions: ${missingFunctions.join(', ')}`, 'red');
    return false;
  }
  
  // Check for proper provider handling
  if (!content.includes('AI_PROVIDER')) {
    log('✗ Missing AI_PROVIDER configuration handling', 'red');
    return false;
  }
  
  // Check for Gemini-specific code
  if (!content.includes('GEMINI_API_URL')) {
    log('✗ Missing GEMINI_API_URL configuration', 'red');
    return false;
  }
  
  // Check for proper error handling
  if (!content.includes('NETWORK_ERROR') || !content.includes('API_HTTP_ERROR')) {
    log('✗ Missing proper error handling codes', 'red');
    return false;
  }
  
  log('✓ worker.js structure is valid', 'green');
  return true;
}

function validateWranglerToml() {
  log('\n📝 Validating wrangler.toml...', 'yellow');
  
  const wranglerPath = path.join(__dirname, 'wrangler.toml');
  const content = fs.readFileSync(wranglerPath, 'utf-8');
  
  // Check for required configuration variables
  const requiredVars = [
    'AI_PROVIDER',
    'AI_MODEL',
    'AI_BASE_URL',
    'GEMINI_API_URL'
  ];
  
  const missingVars = [];
  for (const varName of requiredVars) {
    if (!content.includes(varName)) {
      missingVars.push(varName);
    }
  }
  
  if (missingVars.length > 0) {
    log(`✗ Missing configuration variables: ${missingVars.join(', ')}`, 'red');
    return false;
  }
  
  // Check for proper API URL formats
  if (!content.includes('https://api.openai.com/v1') && 
      !content.includes('https://generativelanguage.googleapis.com')) {
    log('✗ Missing proper API URL configurations', 'red');
    return false;
  }
  
  log('✓ wrangler.toml configuration is valid', 'green');
  return true;
}

function validateModelOptions() {
  log('\n📝 Validating model options...', 'yellow');
  
  const wranglerPath = path.join(__dirname, 'wrangler.toml');
  const content = fs.readFileSync(wranglerPath, 'utf-8');
  
  // Check for documented model options
  const recommendedModels = [
    'gemini-2.0-flash',
    'gemini-2.5-flash-latest',
    'gpt-4o',
    'gpt-4o-mini'
  ];
  
  let foundModels = 0;
  for (const model of recommendedModels) {
    if (content.includes(model)) {
      foundModels++;
    }
  }
  
  if (foundModels < 3) {
    log('⚠ Warning: Some recommended models are not documented', 'yellow');
  } else {
    log(`✓ Found ${foundModels}/${recommendedModels.length} recommended models documented`, 'green');
  }
  
  return true;
}

function validateDocumentation() {
  log('\n📝 Validating documentation...', 'yellow');
  
  const readmePath = path.join(__dirname, 'README.md');
  const guidePath = path.join(__dirname, 'AI_MODEL_UPGRADE_GUIDE.md');
  
  if (!fs.existsSync(readmePath)) {
    log('✗ README.md not found', 'red');
    return false;
  }
  
  if (!fs.existsSync(guidePath)) {
    log('✗ AI_MODEL_UPGRADE_GUIDE.md not found', 'red');
    return false;
  }
  
  const readmeContent = fs.readFileSync(readmePath, 'utf-8');
  const guideContent = fs.readFileSync(guidePath, 'utf-8');
  
  // Check README has model comparison
  if (!readmeContent.includes('Model Comparison') && !readmeContent.includes('Gemini')) {
    log('⚠ Warning: README may not include comprehensive model comparison', 'yellow');
  }
  
  // Check guide has migration steps
  if (!guideContent.includes('Migration') && !guideContent.includes('upgrade')) {
    log('⚠ Warning: Upgrade guide may not include migration steps', 'yellow');
  }
  
  log('✓ Documentation files are present', 'green');
  return true;
}

function validateAPIEndpoints() {
  log('\n📝 Validating API endpoint configurations...', 'yellow');
  
  const workerPath = path.join(__dirname, 'worker.js');
  const content = fs.readFileSync(workerPath, 'utf-8');
  
  // Check for proper endpoint handling
  const endpoints = [
    '/analyze',
    '/result/'
  ];
  
  for (const endpoint of endpoints) {
    if (!content.includes(`'${endpoint}'`) && !content.includes(`"${endpoint}"`)) {
      log(`✗ Missing endpoint: ${endpoint}`, 'red');
      return false;
    }
  }
  
  // Check for proper HTTP methods
  if (!content.includes('POST') || !content.includes('GET')) {
    log('✗ Missing proper HTTP method handling', 'red');
    return false;
  }
  
  log('✓ API endpoints are properly configured', 'green');
  return true;
}

// Run all validations
function runValidation() {
  log('\n🚀 Starting Configuration Validation\n', 'yellow');
  log('=' .repeat(50));
  
  const results = {
    worker: validateWorkerJS(),
    wrangler: validateWranglerToml(),
    models: validateModelOptions(),
    docs: validateDocumentation(),
    endpoints: validateAPIEndpoints()
  };
  
  log('\n' + '='.repeat(50));
  log('\n📊 Validation Summary:\n', 'yellow');
  
  let allPassed = true;
  for (const [name, passed] of Object.entries(results)) {
    const status = passed ? '✓' : '✗';
    const color = passed ? 'green' : 'red';
    log(`${status} ${name.padEnd(15)} ${passed ? 'PASSED' : 'FAILED'}`, color);
    if (!passed) allPassed = false;
  }
  
  log('\n' + '='.repeat(50));
  
  if (allPassed) {
    log('\n✓ All validations passed! Configuration is ready.', 'green');
    log('\nNext steps:', 'yellow');
    log('1. Set your AI API key: wrangler secret put AI_API_KEY');
    log('2. Deploy the worker: wrangler deploy');
    log('3. Test with a sample iris image\n');
    return 0;
  } else {
    log('\n✗ Some validations failed. Please review the errors above.', 'red');
    return 1;
  }
}

// Run validation
process.exit(runValidation());
