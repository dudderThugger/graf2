//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szelestey Ádám
// Neptun : N2BPHM
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
	float f; // function value
	T d;  // derivatives
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f)*g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f)*g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f)*g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f)*g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 300;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extrinsic
	vec3 axis;
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
	} 
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			                                       u.y, v.y, w.y, 0,
			                                       u.z, v.z, w.z, 0,
			                                       0,   0,   0,   1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
			        0,                      1 / tan(fov / 2), 0,                      0,
			        0,                      0,                -(fp + bp) / (bp - fp), -1,
			        0,                      0,                -2 * fp*bp / (bp - fp),  0);
	}

	void rotateCamera(float t) {
		float angle = (t * 2 * M_PI) / 20.0;
		axis = normalize(axis);
		vec4 homogen = vec4(wEye.x, wEye.y, wEye.z, 1) * RotationMatrix(angle, axis);
		wEye = vec3(homogen.x, homogen.y, homogen.z);
		float cosViewAngle = dot(normalize(wEye - wLookat), axis);
		wVup = normalize(wEye - axis * cosViewAngle);
	}

	void calculateVup() {
		float cosViewAngle = dot(normalize(wEye - wLookat), axis);
		wVup  = normalize(wEye - axis * cosViewAngle );
	}
};

//---------------------------
struct Material {
//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
//---------------------------
	vec3 La, Le;
	vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

//---------------------------
class CheckerBoardTexture : public Texture {
//---------------------------
public:
	CheckerBoardTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
class TerrainTexture : public Texture {
//---------------------------
public:
	TerrainTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 brown(0.8, 0.4, 0, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = brown;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	Texture *          texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
//---------------borwn------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//---------------------------
class GouraudShader : public Shader {
//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
		uniform Light[8] lights;     // light source direction 
		uniform int   nLights;		 // number of light sources
		uniform vec3  wEye;          // pos of eye
		uniform Material  material;  // diffuse, specular, ambient ref

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 radiance;		    // reflected radiance

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;	
			vec3 V = normalize(wEye * wPos.w - wPos.xyz);
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
	
			radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
				vec3 H = normalize(L + V);
				float cosd = max(dot(N,H), 0);
				radiance += (vtxPos.z < 0) ? ((-1.0f / (20 * vtxPos.z)) * vec3(0.4, 0.8, 0) + material.ks * pow(cosd, material.shininess)) * lights[i].Le : ((1.0f / (20 * vtxPos.z)) * vec3(0.8, 0.3, 0) + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		in  vec3 radiance;      // interpolated radiance
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	GouraudShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};


//---------------------------
class PhongShader : public Shader {
//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class Geometry {
//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
class ParamSurface : public Geometry {//---------------------------
	
public:
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};
	float t;
	unsigned int nVtxPerStrip, nStrips;
	//ParamSurface(float t) :t(t) { nVtxPerStrip = nStrips = 0; }

	virtual VertexData GenVertexData(float u, float v) = 0;
	//virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

class Body : public Geometry {
public:
	vec3 axis;

	Body() { init(); }

	void init() {
		float vtxs[] = {
			0, 0, 0,
			0, 0.5, 0,
			0.5, 0, 0,
			0.5, 0.5, 0,
			0.5, 0, -1,
			0.5, 0.5, -1,
			0, 0.5, -1,
			0, 0, -1
		};

		unsigned int ele[] = {
			0, 1, 2,
			1, 2, 3,
			0, 1, 6,
			0, 6, 7,
			2, 3, 4,
			0, 2, 7,
			2, 4, 7,
			1, 3, 6,
			3, 5, 6,
			4, 5, 6,
			4, 6, 7
		};

		unsigned int ebo;
		glGenBuffers(1, &ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, sizeof(vtxs), &vtxs[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ele), &ele[0], GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	}
};

class Terrain : public ParamSurface {
	int n; // number of cosinus signals
	std::vector<float> f, phase; // amplitudo, frequency and phase for each cosinus signal

public:
	Terrain(int numberOfSigns) :n(numberOfSigns) { 
		create();
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		float x = asin(u); float y = acos(v) - M_PI_2;
		// generate random values for each cosinus signal
		for(int i = 0; i < n; ++i) {
			f.push_back((float)(rand() % 50));
			phase.push_back((M_PI * (rand() % 100)) / (100.0f));
			//printf("%f %f\n", f[i], M_PI * (rand() % 20) / (20.0f));
		}
		// generate vertex data
		float height = 0;
		float xd = 0;
		float yd = 0;
		//printf("x: %f, y: %f\n", x, y);
		for(int i = 0; i < n; ++i) {
			for(int j = 0 ; j < n; ++j) {
				float A = 0.15 / sqrtf(f[i] * f[i] + f[j] * f[j]);
				float angle = x * f[i] + y * f[j] + phase[i] * phase[j];
				height += A * cos(angle);
				xd += f[i] * A * sin(angle);
				yd += f[j] * A * sin(angle);
			}
		}
		//printf("h: %f, xd: %f, yd: %f\n", height, xd, yd);
		vtxData.position = vec3(u, v, height);
		vec3 normPos = normalize(vtxData.position);
		vtxData.normal = normalize(vec3(xd, yd, 1));
		return vtxData;
	}
};

//---------------------------
struct Object {
//---------------------------
	Shader *   shader;
	Material * material;
	Texture *  texture;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend; }
};

float a = 0.2;
float b = 0.3;
float c = 0.4;
const float g = 9.8;
const float mass = 1;
float l0 = 0.4;
const float d = 10; // rugoallando
vec3 s = vec3(0, 0, 6);
vec3 bodyMass = vec3(1, 0, 6);
vec3 velocity = vec3(0, 0, 0);
vec3 oriantation = vec3(1, 0, 0);
float I = mass * (a*a + b*b) / 12;
float w = 0;

void calculateBody(float dt) {
	// ma = m*q + K + pv
	vec3 K = d * normalize(s - bodyMass) * (sqrt(dot(s - bodyMass, s - bodyMass)) - l0);

	vec3 a = (vec3(0, 0, -1) * g + K) / mass;
	velocity = velocity + a * dt;
	bodyMass = bodyMass + 0.5 * a * dt * dt + velocity * dt;
}

//---------------------------
class Scene {
//---------------------------
	std::vector<Object *> objects;
	Camera camera1, camera2; // 3D camera
	std::vector<Light> lights;
	Object* bodyObject;
public:
	void Build() {
		// Shaders
		Shader * terrainShader = new GouraudShader();
		Shader* phong = new PhongShader();

		// Textures
		Texture* terrainTexture = new TerrainTexture(1, 1);
		Texture* chekerBoardTexture = new CheckerBoardTexture(1, 1);

		Material * terrainMaterial = new Material;
		terrainMaterial->kd = vec3(0.8f, 0.8f, 0.0f);
		terrainMaterial->ks = vec3(0.2f, 0.2f, 0.2f);
		terrainMaterial->ka = vec3(0.0, 0.0, 0.0);
		terrainMaterial->shininess = 5;

		Material * bodyMaterial = new Material;
		bodyMaterial->kd = vec3(0.8f, 0.4f, 0.0f);
		bodyMaterial->ks = vec3(0.2f, 0.2f, 0.2f);
		bodyMaterial->ka = vec3(0.5, 0.5, 0.5);
		bodyMaterial->shininess = 50;

		// Geometries
		Geometry* terrain = new Terrain(12);
		Body* body = new Body();

		// Objects
		Object* terrainObject = new Object(terrainShader, terrainMaterial, terrainTexture, terrain);
		terrainObject->translation = vec3(5, -5, 2);
		terrainObject->rotationAxis = vec3(0, 0, 1);
		terrainObject->rotationAngle = M_PI / 2;
		terrainObject->scale = vec3(10, 10, 10);
		objects.push_back(terrainObject);

		bodyObject = new Object(phong, bodyMaterial, chekerBoardTexture, body);
		bodyObject->translation = bodyMass;
		bodyObject->scale = vec3(a, b, c);
		objects.push_back(bodyObject);

		// Cameras
		camera1.wEye = vec3(0, -3, 6);
		camera1.wLookat = vec3(0, 0, 1);
		camera1.wVup = vec3(0, 1, 0);
		camera1.axis = vec3(0, 0, 1);
		camera1.calculateVup();

		camera2.wEye = vec3(0, 0, 4);
		camera2.wLookat = vec3(0, 0, 1);
		camera2.wVup = vec3(0, 1, 0);
		

		// Lights
		lights.resize(1);
		lights[0].wLightPos = vec4(0, 0, 20, 0);	// ideal point -> directional light source
		lights[0].La = vec3(0.7, 0.7, 0.7);
		lights[0].Le = vec3(0.7f, 0.7f, 0.7f);
	}

	void Render1() {
		RenderState state;
		state.wEye = camera1.wEye;
		state.V = camera1.V();
		state.P = camera1.P();
		state.lights = lights;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Render2() {
		RenderState state;
		state.wEye = camera2.wEye;
		state.V = camera2.V();
		state.P = camera2.P();
		state.lights = lights;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(float dt) {
		rotateCamera(dt);
		calculateBody(dt);
		bodyObject->translation = bodyMass;
		camera2.wEye = bodyMass + a;
	}

	void rotateCamera(float dt) {
		camera1.rotateCamera(dt);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, windowWidth/2, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	
	scene.Render2();

	glViewport(windowWidth/2, 0, windowWidth/2, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	
	scene.Render1();



	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is �infinitesimal�
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(Dt);
	}
	glutPostRedisplay();
}
