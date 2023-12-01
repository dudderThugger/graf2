//************************************************************************* //
// Game Engine
//************************************************************************* //
#include "gameengine.h"

Input input;
const float GAME_SPEED = 1000.0f;

//-----------------------------------------------------------------
void onDisplay(void) {
//-----------------------------------------------------------------
	glClearColor(0, 0, 0, 0);          
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
	scene->Render();
	glutSwapBuffers( );
}   

//-----------------------------------------------------------------
void onIdle(void) { 
//-----------------------------------------------------------------
	static float tend = 0;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / GAME_SPEED;
	scene->Simulate(tstart, tend);
	glutPostRedisplay();
}

//-----------------------------------------------------------------
void onKeyboard(unsigned char key, int x, int y) { // ascii chars
//-----------------------------------------------------------------
    input.glutKeyTable[key] = true;
}

//-----------------------------------------------------------------
void onKeyboardUp(unsigned char key, int x, int y) { // ascii chars
//-----------------------------------------------------------------
    input.glutKeyTable[key] = false;
}

//-----------------------------------------------------------------
void onSpecialKey(int key, int x, int y) { // not ascii chars
//-----------------------------------------------------------------
	input.glutKeyTable[key] = true;
}

//-----------------------------------------------------------------
void onSpecialKeyUp(int key, int x, int y) { // not ascii chars
//-----------------------------------------------------------------
	input.glutKeyTable[key] = false;
}

Scene  * scene;

Geometry * BillBoard::geometry = NULL;
Shader *   BillBoard::shader = NULL;
Shader *   ParticleSystem::shader = NULL;
Texture *  ParticleSystem::texture = NULL;

// Initialization, create an OpenGL context
//-----------------------------------------------------------------
void onInitialization() {
//-----------------------------------------------------------------
	BillBoard::shader = ParticleSystem::shader = new VolumetricShader();
	BillBoard::geometry = new Quad();
	ParticleSystem::texture = new Texture("explosion.bmp", true);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene = new Scene;
	scene->Build();
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Entry point of the application
int main(int argc, char * argv[]) {
	// Initialize GLUT, Glew and OpenGL 
	glutInit(&argc, argv);

	// OpenGL major and minor versions
	int majorVersion = 3, minorVersion = 3;
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif
	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	// Initialize this program and create shaders
	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutSpecialFunc(onSpecialKey);
	glutSpecialUpFunc(onSpecialKeyUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	return 1;
}
